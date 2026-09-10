# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
from logging import getLogger
from typing import AsyncIterator, Callable

import numpy as np
import torch

from ..models import LMGen, MimiModel, loaders
from .stt import STTMarkerMessage, STTWordMessage


logger = getLogger(__name__)

# Match ``stt.py`` / Gradium: ignore unstable VAD for the first few steps after audio starts.
_VAD_SKIP_STEPS = 12


class LocalSpeechToText:
    """Wrapper for local STT using the same Mimi + LMGen stack based on repo kyutai/stt-1b-en_fr-candle"""

    def __init__(
        self,
        mimi: MimiModel,
        hf_repo: str = "kyutai/stt-1b-en_fr-candle",
        device: str | torch.device | None = None,
        vad_callback: Callable[[float], None] | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.vad_callback = vad_callback

        if device is None:
            try:
                self._device = next(mimi.parameters()).device
            except StopIteration:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device) if isinstance(device, str) else device

        # Set up STT models and mimi
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(hf_repo)
        self.text_tokenizer = checkpoint_info.get_text_tokenizer()
        lm = checkpoint_info.get_moshi(device=self._device, dtype=dtype)
        self._lm_gen = LMGen(lm, cfg_coef=1.0, **checkpoint_info.lm_gen_config)
        self._prime_cap = max(self._lm_gen.lm_model.delays)

        self.mimi = mimi
        self.mimi.set_num_codebooks(self._lm_gen.lm_model.num_codebooks - 1)
        self.mimi.streaming_forever(1)
        self._lm_gen.streaming_forever(1)

        # Communication with the batched step loop.
        self._out_queue: asyncio.Queue[STTWordMessage | STTMarkerMessage | None] = asyncio.Queue()
        self._lock = asyncio.Lock()

        # Shutdown event
        self.shutdown_complete = asyncio.Event()
        self.shutdown_complete.set()

        # State variables
        self.sent_samples = 0
        self._pending = np.zeros(0, dtype=np.float32)
        self._frame_id = 0
        self._playhead_s = 0.0
        self._vad_step_count = 0

    def _decode_user_token(self, text_token: int) -> str | None:
        if text_token in (0, 1, 2, 3):
            return None
        text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore[arg-type]
        return str(text)

    @torch.inference_mode()
    def _run_codes(self, codes: torch.Tensor, decoder: LMGen) -> STTWordMessage | None:
        assert self._lm_gen is not None and self.mimi is not None

        mask = torch.ones(1, device=self._device, dtype=torch.bool)
        self._lm_gen.set_exec_mask(mask)

        while self._frame_id < self._prime_cap:
            _ = self._lm_gen.step_with_extra_heads(codes)
            self._frame_id += 1

        result = self._lm_gen.step_with_extra_heads(codes)
        assert result is not None
        tokens, vad_heads = result
        token = int(tokens[0, 0].cpu().item())

        if self.vad_callback and vad_heads and len(vad_heads) > 2:
            if self._vad_step_count >= _VAD_SKIP_STEPS:
                vad_value = vad_heads[2][0, 0, 0].cpu().item()
                self.vad_callback(float(vad_value))
            self._vad_step_count += 1

        decoded = self._decode_user_token(token)
        self._frame_id += 1

        if decoded is None or decoded.strip() == "":
            return None
        decoded = decoded.replace("▁", " ")
        return STTWordMessage(type="Word", text=decoded, start_time=self._playhead_s, id=token)

    async def send_audio(self, audio: np.ndarray) -> None:
        if audio.ndim != 1:
            raise ValueError(f"Expected 1D array, got {audio.shape=}")
        if audio.dtype != np.float32:
            raise ValueError(f"Expected float32 array, got {audio.dtype=}")

        self.sent_samples += len(audio)

        async with self._lock:
            if self._pending.size:
                self._pending = np.concatenate([self._pending, audio])
            else:
                self._pending = audio

            fs = int(self.mimi.sample_rate / self.mimi.frame_rate)
            sr = int(self.mimi.sample_rate)
            while self._pending.size >= fs:
                frame = self._pending[:fs].copy()
                self._pending = self._pending[fs:]

                chunk = torch.from_numpy(frame).to(device=self._device, dtype=torch.float32).view(1, 1, -1)
                self.mimi.set_exec_mask(torch.ones(1, device=self._device, dtype=torch.bool))
                codes = self.mimi.encode(chunk)
                codes = codes[:, : self._lm_gen.needed_tokens].to(self._device)
                assert codes.shape[-1] == 1

                word = self._run_codes(codes, self._lm_gen)
                if word is not None:
                    await self._out_queue.put(word)
                self._playhead_s += float(fs) / float(sr)

    async def flush(self) -> None:
        return

    async def start_up(self) -> None:
        self.shutdown_complete.clear()
        self._out_queue = asyncio.Queue()

        mask = torch.ones(1, device=self._device, dtype=torch.bool)
        with torch.inference_mode():
            self.mimi.reset_streaming(mask)
            self._lm_gen.reset_streaming(mask)

        self.sent_samples = 0
        self._pending = np.zeros(0, dtype=np.float32)
        self._frame_id = 0
        self._playhead_s = 0.0
        self._vad_step_count = 0

        logger.info("LocalSpeechToText session started")

    async def shutdown(self) -> None:
        logger.info("Shutting down LocalSpeechToText")

        await self._out_queue.put(None)

        if not self.shutdown_complete.is_set():
            try:
                await asyncio.wait_for(self.shutdown_complete.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("LocalSpeechToText shutdown timed out waiting for iterator completion")
        self.shutdown_complete.set()

        logger.info("LocalSpeechToText shutdown() finished")

    async def __aiter__(self) -> AsyncIterator[STTWordMessage | STTMarkerMessage]:
        while True:
            msg = await self._out_queue.get()
            if msg is None:
                self.shutdown_complete.set()
                break
            yield msg
