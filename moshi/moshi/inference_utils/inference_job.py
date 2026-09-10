# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the LICENSE file in the root directory of this source tree.

"""Per-audio-file offline inference session (queues, STT, RAG, trace) for ``OfflineServerState``."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import time
import wave
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import sphn
import torch

from .audio_processor import AudioProcessor
from .channel import StepInput, StepOutput
from .rag_manager import RAGManager
from .turn_manager import TurnManager
from .utils import get_conditioning_remote_async
from ..stt import LocalSpeechToText, STTWordMessage
from ..server import ServerState

logger = logging.getLogger(__name__)


def _load_audio_mono_float32(path: Path, target_sr: int) -> np.ndarray:
    """Load audio as mono float32 in ``[-1, 1]`` at ``target_sr``."""
    pcm, out_sr = sphn.read(str(path), sample_rate=target_sr)
    if pcm.ndim != 2:
        raise ValueError(f"Expected 2D audio from sphn.read, got shape {pcm.shape} for {path}")
    if out_sr != target_sr:
        logger.warning(
            "sphn.read returned sample_rate %s != requested %s for %s; using returned audio as-is",
            out_sr,
            target_sr,
            path,
        )
    # Channel dimension first: [C, T] -> mono
    x = np.mean(pcm.astype(np.float64), axis=0).astype(np.float32)
    np.clip(x, -1.0, 1.0, out=x)
    return x


def _write_wav_mono_float32(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    """Write mono float32 PCM in ``[-1, 1]`` as 16-bit WAV."""
    pcm = np.clip(pcm.astype(np.float32), -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())


class InferenceJob:
    """One file, one batch slot: feed PCM, consume LM outputs, write trace JSON and model WAV."""

    def __init__(
        self,
        server: ServerState,
        wav_path: Path,
        output_path: Path,
        stop_on_end_of_input: bool,
        use_gt_reference: bool,
        max_tail_silence: int | None,
        sidecar: dict[str, Any],
        stt: LocalSpeechToText | None = None,
    ):
        self.server = server
        self.wav_path = wav_path
        self.output_path = output_path
        self.stop_on_end_of_input = stop_on_end_of_input
        self.max_tail_silence = max_tail_silence

        self.turn_manager = TurnManager(
            window_size=server.vad_window_size,
            threshold=server.vad_threshold,
            stt_wait_steps=server.stt_wait_steps,
            init_active_speaker=server.init_active_speaker,
        )
        self.rag_manager = RAGManager(
            reference_generator=server.reference_generator,
            rag_timeout=server.rag_timeout,
            max_tokens=server.max_reference_tokens,
            gt_reference_text=sidecar.get("gt_reference_text", None) if use_gt_reference else None,
        )
        self.audio_processor = AudioProcessor(power_threshold=server.power_threshold)

        self.stt = stt

        # Communication with the batched step loop.
        self.input_queue: asyncio.Queue[StepInput] = asyncio.Queue()
        self.output_queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        self.slot_idx = -1

        # Per-job state and output trace
        self.step_index = 0
        self.model_text: list[str] = []
        self.user_text: list[str] = []

        self.trace: dict[str, Any] = {
            "rag_trigger_step": -1,
            "question_end_step": -1,
            "retrieval_step": -1,
            "conditioning_step": -1,
            "reference_text": "",
            "model_text": self.model_text,
            "user_text": self.user_text,
            "gt_user_text": sidecar.get("gt_user_text"),
            "gt_reference_text": sidecar.get("gt_reference_text"),
            "answer": sidecar.get("answer"),
        }

        # Communication between async loops
        self._doing_retrieval = False
        self._retrieval_start_time: float | None = None
        self._retrieval_start_step: str | None = None
        self._retrieval_done_step: int | None = None
        self._user_id_buffer: deque[int] = deque()
        self._pcm_one_step_cv = asyncio.Condition()
        self._done = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        self._feed_finished = asyncio.Event()
        self._model_pcm_chunks: list[np.ndarray] = []

    def _decode_text_token(self, text_token: int) -> str | None:
        if text_token in (0, 1, 2, 3):
            return None
        text = self.server.text_tokenizer.id_to_piece(text_token)  # type: ignore[arg-type]
        text = text.replace("▁", " ")
        if text_token == self.server.runner.lm_gen.lm_model.rag_token_id:
            return "[RET]"
        return text

    def _check_tail_silence(self) -> bool:
        consecutive = 0
        for i, piece in enumerate(self.model_text):
            if i < self.trace.get("question_end_step", -1):
                continue
            if piece == "<pad>":
                consecutive += 1
                if consecutive > self.max_tail_silence:
                    return True
            else:
                consecutive = 0
        return False

    async def _async_update_reference(self, reference_text: str) -> None:
        streaming_sum_tensor = await get_conditioning_remote_async(
            text=reference_text,
            encoder_url=self.server.reference_encoder_url,
        )
        batch_size = self.server.batch_size
        per_slot: list[torch.Tensor | None] = [None] * batch_size
        per_slot[self.slot_idx] = streaming_sum_tensor.squeeze(0)
        self.server.runner.lm_gen.update_streaming_sum_tensors(per_slot)
        self.trace["conditioning_step"] = self.step_index

    async def _handle_reference_text(self, reference_text: str | None, lm_label: str = "") -> None:
        await self._async_update_reference(reference_text or "")
        self._retrieval_start_time = None
        self._retrieval_start_step = None
        self._retrieval_done_step = None

    async def _catch_reference_text(self, reference_text: str | None, lm_label: str = "") -> None:
        self.trace["reference_text"] = reference_text or ""
        assert self._retrieval_start_time is not None
        retrieval_elapsed = max(0.0, time.monotonic() - self._retrieval_start_time)

        frame_rate = float(self.server.runner.mimi.frame_rate)
        retrieval_steps = math.floor(retrieval_elapsed * frame_rate) if frame_rate > 0 else 0
        self._retrieval_done_step = self.step_index + retrieval_steps
        self._doing_retrieval = False

    async def _stt_recv_loop(self) -> None:
        assert self.stt is not None

        async def _consume_stt() -> None:
            async for msg in self.stt:
                if isinstance(msg, STTWordMessage):
                    ut = msg.text.replace("▁", " ")
                    uid = msg.id
                    self.turn_manager.handle_spoken_text(user_text=ut)
                    self._user_id_buffer.append(uid)

        consumer = asyncio.create_task(_consume_stt())
        try:
            await self._shutdown_event.wait()
        finally:
            consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consumer

    async def _wait_step_index_at_least(self, min_step: int) -> bool:
        """Block until ``step_index >= min_step`` or shutdown."""
        async with self._pcm_one_step_cv:
            while self.step_index < min_step:
                if self._shutdown_event.is_set():
                    return False
                await self._pcm_one_step_cv.wait()
        return not self._shutdown_event.is_set()

    async def _wait_until(self, target_time: float) -> bool:
        """Block until ``time.monotonic() >= target_time`` or shutdown."""
        now = time.monotonic()
        wait_s = target_time - now
        if wait_s > 0:
            await asyncio.sleep(wait_s)

    async def _feed_loop(self) -> None:
        device = self.server.device
        sr = int(self.server.runner.mimi.sample_rate)
        pcm = _load_audio_mono_float32(self.wav_path, sr)
        frame = self.server.frame_size
        tail = len(pcm) % frame
        if tail:
            pcm = pcm[: len(pcm) - tail]
        pcm_offsets = list(range(0, len(pcm), frame))
        max_stream_delay = max(self.server.runner.lm_gen.delays_cuda).item()
        for feed_step, off in enumerate(pcm_offsets):
            if feed_step > 0 and not await self._wait_step_index_at_least(feed_step - max_stream_delay):
                return
            if self._shutdown_event.is_set():
                return
            chunk_np = pcm[off : off + frame]
            chunk = torch.from_numpy(chunk_np).to(device=device, dtype=torch.float32)[None, None]
            filtered = self.audio_processor.filter_by_power(chunk)
            await self.input_queue.put(
                StepInput(filtered_pcm=filtered[0], is_first=off == 0, pcm=chunk[0]),
            )
            await self.stt.send_audio(chunk_np.astype(np.float32))
            self.trace["question_end_step"] = feed_step

        if self.stop_on_end_of_input:
            try:
                await self.stt.flush()
            except Exception as e:
                logger.debug("stt flush: %s", e)
            self._feed_finished.set()
            return

        while not self._shutdown_event.is_set():
            if feed_step > 0 and not await self._wait_step_index_at_least(feed_step - max_stream_delay):
                return
            if self._shutdown_event.is_set():
                return
            z = torch.zeros(1, 1, frame, dtype=torch.float32, device=device)
            await self.input_queue.put(StepInput(filtered_pcm=z[0], is_first=False, pcm=z[0]))
            await self.stt.send_audio(np.zeros(frame, dtype=np.float32))
            feed_step += 1

    async def _output_loop(self) -> None:
        assert self._task_group is not None
        while not self._shutdown_event.is_set():
            # Do not forward model until the retrieval is complete.
            while self._doing_retrieval:
                await asyncio.sleep(0.05)
                if self._shutdown_event.is_set():
                    return
                if not self._doing_retrieval:
                    break
            if self.stop_on_end_of_input and self._feed_finished.is_set():
                try:
                    out = await asyncio.wait_for(self.output_queue.get(), timeout=1.0)
                except TimeoutError:
                    await self._finalize()
                    return
            else:
                out = await self.output_queue.get()

            if out.pcm is not None:
                self._model_pcm_chunks.append(out.pcm.detach().cpu().float().numpy().reshape(-1))

            text_token = out.text_token

            if text_token == self.server.runner.lm_gen.lm_model.rag_token_id:
                self.trace["rag_trigger_step"] = self.step_index
                self._retrieval_start_step = self.step_index + int(self.turn_manager.stt_wait_steps)
                self.trace["retrieval_step"] = self._retrieval_start_step
                self.model_text.append(self.server.text_tokenizer.id_to_piece(text_token))  # type: ignore[arg-type]
                await self.rag_manager.trigger(
                    task_group=self._task_group,
                    wait_steps=int(self.turn_manager.stt_wait_steps),
                    handle_reference_fn=self._catch_reference_text,
                    context_provider=self.turn_manager.get_context,
                )
            else:
                decoded = self._decode_text_token(text_token)
                self.turn_manager.handle_spoken_text(model_text=decoded)
                if decoded is None:
                    self.model_text.append("<pad>")
                else:
                    self.model_text.append(self.server.text_tokenizer.id_to_piece(text_token))  # type: ignore[arg-type]

            if self._user_id_buffer:
                uid = self._user_id_buffer.popleft()
                self.user_text.append(self.stt.text_tokenizer.id_to_piece(uid))  # type: ignore[arg-type]
            else:
                self.user_text.append("<pad>")

            self.rag_manager.step()
            if self.step_index == self._retrieval_start_step:
                # Retrieval start time is after the wait steps ends instead of right after the RAG trigger step.
                self._retrieval_start_time = time.monotonic()
                self._doing_retrieval = True
            if self._retrieval_done_step is not None and self.step_index >= self._retrieval_done_step:
                await self._handle_reference_text(self.trace["reference_text"])

            async with self._pcm_one_step_cv:
                self.step_index += 1
                self._pcm_one_step_cv.notify_all()

            if self.trace.get("question_end_step", -1) >= 0 and self.max_tail_silence is not None:
                if self._check_tail_silence():
                    await self._finalize()
                    return

    async def _finalize(self) -> None:
        self._shutdown_event.set()
        async with self._pcm_one_step_cv:
            self._pcm_one_step_cv.notify_all()
        self.rag_manager.cancel_pending()
        self.trace["model_text"] = self.model_text
        self.trace["user_text"] = self.user_text
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(self.trace, indent=2), encoding="utf-8")
        if self._model_pcm_chunks:
            wav_path = self.output_path.with_suffix(".wav")
            sr = int(self.server.runner.mimi.sample_rate)
            pcm_all = np.concatenate(self._model_pcm_chunks)
            _write_wav_mono_float32(wav_path, pcm_all, sr)
        self._done.set()

    async def run(self, rag_task_group: asyncio.TaskGroup) -> None:
        """Run feed, STT recv, and output loops; ``rag_task_group`` must stay open for RAG background tasks."""
        self._task_group = rag_task_group
        async with self.rag_manager:
            await self.stt.start_up()
            setattr(self.stt, "vad_callback", self.turn_manager.update_vad)
            try:
                await asyncio.gather(
                    self._feed_loop(),
                    self._stt_recv_loop(),
                    self._output_loop(),
                )
            finally:
                await self.stt.shutdown()

    async def wait_done(self) -> None:
        await self._done.wait()
