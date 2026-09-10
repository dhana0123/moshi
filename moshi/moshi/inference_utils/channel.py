# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Per-channel (per-client) state for a moshi conversation.

A ``Channel`` owns everything that is specific to a single connected user: the
WebSocket, the opus reader/writer, the running PCM buffer, the turn manager,
the per-channel RAG manager (with its own reference history) and the STT
client. The shared GPU resources (mimi, lm_gen, the LLM reference generator)
live on the ``ServerState`` and run inside a single batched step loop.

Audio I/O between a channel and the batched step loop is queue-based:
    Channel ──pcm frames──> input_queue ──┐
                                          │  (server batched step)
    Channel <──text/pcm─── output_queue <─┘

Use as an async context manager:

    async with Channel(server, ws) as channel:
        await channel.run()
"""

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass

import aiohttp
import numpy as np
import sphn
import torch
import websockets
from aiohttp import web

from .audio_processor import AudioProcessor
from .rag_manager import RAGManager
from .turn_manager import TurnManager
from .utils import get_conditioning_remote_async
from ..stt import GradiumSpeechToText, LocalSpeechToText, STTWordMessage
from ..models import MimiModel

logger = logging.getLogger(__name__)


class _ChannelClosed(Exception):
    """Raised by ``_recv_loop`` when the WebSocket connection is closed.

    This is not an error — it is the normal shutdown signal.  The ``TaskGroup``
    in ``Channel.run()`` uses it to cancel sibling tasks cleanly.
    """


@dataclass
class StepInput:
    """One frame of audio from a channel to the batched step loop."""

    filtered_pcm: torch.Tensor  # shape [1, frame_size], power-threshold filtered for main Mimi
    is_first: bool = False  # True for the first frame of the session
    # Raw frame for in-process STT; when ``None``, local STT uses ``filtered_pcm`` (live server omits this).
    pcm: torch.Tensor | None = None


@dataclass
class StepOutput:
    """One frame of model output produced for a channel by the step loop."""

    text_token: int
    pcm: torch.Tensor | None  # shape [frame_size], cpu; None during acoustic delay


class Channel:
    """Per-user state and event loops for a single moshi conversation."""

    # Special token used to refresh the reference text in the UI.
    _REFRESH_TOKEN = "\0"

    def __init__(self, server, ws: web.WebSocketResponse, mimi: MimiModel | None = None):
        self.server = server
        self.ws = ws

        self.opus_writer = sphn.OpusStreamWriter(server.runner.mimi.sample_rate)
        self.opus_reader = sphn.OpusStreamReader(server.runner.mimi.sample_rate)

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
        )
        self.audio_processor = AudioProcessor(power_threshold=server.power_threshold)

        if server.gradium_stt:
            self.stt = GradiumSpeechToText(
                expected_language="en",
                vad_callback=self.turn_manager.update_vad,
            )
        else:
            assert mimi is not None
            self.stt = LocalSpeechToText(mimi=mimi, vad_callback=self.turn_manager.update_vad)

        # Communication with the batched step loop.
        self.input_queue: asyncio.Queue[StepInput] = asyncio.Queue()
        self.output_queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        self.slot_idx: int = -1
        self._is_first_frame: bool = True

        self._all_pcm_data: np.ndarray | None = None
        self._stack: contextlib.AsyncExitStack | None = None
        self._task_group: asyncio.TaskGroup | None = None
        self._log = logging.LoggerAdapter(logger, extra={})

    async def __aenter__(self) -> "Channel":
        self._stack = contextlib.AsyncExitStack()
        await self._stack.__aenter__()
        # Acquire a batch slot first; if none is free we want to fail fast before
        # touching the heavier resources below.
        self.slot_idx = await self.server.acquire_slot(self)
        self._log = logging.LoggerAdapter(logger, {"slot": self.slot_idx})
        self._log.process = lambda msg, kwargs: (f"[slot {self.slot_idx}] {msg}", kwargs)
        self._stack.push_async_callback(self.server.release_slot, self.slot_idx)
        await self._stack.enter_async_context(self.rag_manager)
        await self.stt.start_up()
        self._stack.push_async_callback(self.stt.shutdown)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        assert self._stack is not None
        try:
            return await self._stack.__aexit__(exc_type, exc, tb)
        finally:
            self._stack = None
            self._log.info("WebSocket connection closed")

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    async def _send_turn_outputs(self, outputs: list[tuple[str, str]]):
        for text, role in outputs:
            await self._encode_and_send_message(text, role=role)

    async def _encode_and_send_message(self, text: str, type: str = "text", role: str = "model"):
        assert type in ("text", "referencetext")
        assert role in ("model", "user")
        color_id = 4 if role == "model" else 10
        type_bit = b"\x07" if type == "text" else b"\x09"
        msg = type_bit + bytes([color_id]) + bytes(text, encoding="utf8")
        await self.ws.send_bytes(msg)

    # ------------------------------------------------------------------
    # Reference-text handling
    # ------------------------------------------------------------------

    async def _handle_reference_text(self, reference_text: str | None, lm_label: str = ""):
        """Forward a freshly generated reference text to the UI and the LM.

        Args:
            reference_text: Generated reference text.
            lm_label: Display name of the LLM used for reference generation (sent to client UI).
        """
        if reference_text is None:
            preview, ref_len = "", 0
        else:
            preview, ref_len = reference_text[:120], len(reference_text)
        self._log.info(f"[Reference] received reference text (len={ref_len}) lm={lm_label!r}: '{preview}'")
        if reference_text:
            await self._encode_and_send_message(self._REFRESH_TOKEN, type="referencetext", role="model")
            # LM id, tab, then reference text (text may contain further tabs).
            await self._encode_and_send_message(f"{lm_label}\t{reference_text}", type="referencetext", role="model")
        elif reference_text == "":
            await self._send_turn_outputs([("[RET_FAILED]", "model")])

        self._log.info("[Reference] requesting ARC encoding for new reference")
        assert self._task_group is not None, "Channel.run() must be active"
        self._task_group.create_task(self._async_update_reference(reference_text or ""))

    async def _async_update_reference(self, reference_text: str):
        request_started = time.time()
        streaming_sum_tensor = await get_conditioning_remote_async(
            text=reference_text,
            encoder_url=self.server.reference_encoder_url,
        )
        self._log.info(
            f"[Reference] ARC encoding received in {time.time() - request_started:.3f}s "
            f"(streaming_sum {tuple(streaming_sum_tensor.shape)})"
        )
        # Build a per-slot update list: only this channel's slot gets the new tensor.
        batch_size = self.server.batch_size
        # streaming_sum_tensor is [1, T, dim] from the remote encoder; squeeze batch dim.
        per_slot: list[torch.Tensor | None] = [None] * batch_size
        per_slot[self.slot_idx] = streaming_sum_tensor.squeeze(0)  # [T, dim]
        self.server.runner.lm_gen.update_streaming_sum_tensors(per_slot)
        self._log.info("[Reference] updated streaming_sum condition on LM")

    def _decode_text_token(self, text_token: int) -> str | None:
        if text_token in (0, 1, 2, 3):
            return None
        text = self.server.text_tokenizer.id_to_piece(text_token)  # type: ignore[arg-type]
        text = text.replace("▁", " ")
        if text_token == self.server.runner.lm_gen.lm_model.rag_token_id:
            return "[RET]"
        return text

    # ------------------------------------------------------------------
    # Main loops
    # ------------------------------------------------------------------

    async def run(self):
        """Run the recv / STT / output loops to completion.

        Uses a ``TaskGroup`` so that any unhandled exception (including from
        dynamically spawned tasks like ARC encoding requests) propagates
        immediately and cancels sibling tasks.

        Normal shutdown: ``_recv_loop`` raises ``_ChannelClosed`` when the
        WebSocket closes, which makes the TaskGroup cancel the sibling loops.
        Connection-related errors from siblings racing with the close are
        suppressed — only truly unexpected errors propagate.
        """
        await self.ws.send_bytes(b"\x00")
        # Send retrieval backend capabilities if multi-profile switching is available.
        retrieval_profiles = getattr(self.server, "_retrieval_profiles", [])
        if len(retrieval_profiles) >= 2:
            from .retrieval_profiles import default_profile_id

            caps = {
                "retrieval_backends": [{"id": p.id} for p in retrieval_profiles],
                "retrieval_backend_default": default_profile_id(retrieval_profiles),
            }
            await self.ws.send_bytes(b"\x04" + json.dumps(caps).encode("utf-8"))
            self.rag_manager.set_retrieval_profile_id(default_profile_id(retrieval_profiles))
        try:
            async with asyncio.TaskGroup() as tg:
                self._task_group = tg
                tg.create_task(self._recv_loop())
                tg.create_task(self._stt_recv_loop())
                tg.create_task(self._output_loop())
        except _ChannelClosed:
            pass  # Normal disconnect — siblings were cancelled by the TaskGroup.
        except (ConnectionError, asyncio.IncompleteReadError) as eg:
            # Connection errors racing with disconnect are expected.
            for exc in eg.exceptions:
                self._log.debug(f"suppressed during shutdown: {exc}")
        finally:
            self._task_group = None

    async def _recv_loop(self):
        """Read PCM from the WebSocket, feed STT, and queue mimi-ready frames.

        Raises ``_ChannelClosed`` when the connection ends so that the
        ``TaskGroup`` in ``run()`` cancels the sibling loops immediately.
        """
        server = self.server
        frame_size = server.frame_size
        async for message in self.ws:
            if message.type == aiohttp.WSMsgType.ERROR:
                self._log.error(f"WebSocket error: {self.ws.exception()}")
                break
            elif message.type == aiohttp.WSMsgType.CLOSED:
                break
            elif message.type != aiohttp.WSMsgType.BINARY:
                self._log.error(f"unexpected WebSocket message type {message.type}")
                continue
            data = message.data
            if not isinstance(data, bytes) or len(data) == 0:
                continue
            kind = data[0]
            if kind == 4:  # metadata (client → server)
                try:
                    payload = json.loads(data[1:].decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._log.warning("invalid metadata json from client")
                    continue
                rid = payload.get("retrieval_backend_id")
                if rid is None:
                    continue
                retrieval_profiles = getattr(self.server, "_retrieval_profiles", [])
                if len(retrieval_profiles) < 2:
                    continue
                allowed = {p.id for p in retrieval_profiles}
                if rid not in allowed:
                    self._log.warning("ignored unknown retrieval_backend_id %r", rid)
                    continue
                self.rag_manager.set_retrieval_profile_id(str(rid))
                self._log.info("retrieval backend switched to %r", rid)
                continue
            if kind != 1:
                self._log.warning(f"unknown binary message kind byte {kind:#x}")
                continue

            payload = data[1:]
            pcm = self.opus_reader.append_bytes(payload)
            if pcm.shape[-1] == 0:
                continue
            buf = pcm if self._all_pcm_data is None else np.concatenate((self._all_pcm_data, pcm))

            while buf.shape[-1] >= frame_size:
                chunk_np = buf[:frame_size]
                buf = buf[frame_size:]
                chunk = torch.from_numpy(chunk_np).to(device=server.device)[None, None]

                try:
                    await self.stt.send_audio(chunk[0, 0].cpu().numpy())
                except websockets.ConnectionClosed:
                    # STT services may enforce a hard max session duration (e.g. 300s)
                    # and close with policy-violation. Treat this as normal channel end.
                    raise _ChannelClosed()

                filtered_chunk = self.audio_processor.filter_by_power(chunk)
                step_input = StepInput(filtered_pcm=filtered_chunk[0], is_first=self._is_first_frame)
                self._is_first_frame = False
                await self.input_queue.put(step_input)

            self._all_pcm_data = buf

        raise _ChannelClosed()

    async def _stt_recv_loop(self):
        async for msg in self.stt:
            if isinstance(msg, STTWordMessage):
                await self._send_turn_outputs(self.turn_manager.handle_spoken_text(user_text=msg.text))

    async def _output_loop(self):
        """Consume model outputs from the batched step loop and forward to the client."""
        while True:
            out = await self.output_queue.get()
            text_token = out.text_token

            # Audio first (latency-critical for the user).
            if out.pcm is not None:
                opus_bytes = self.opus_writer.append_pcm(out.pcm.numpy())
                if len(opus_bytes) > 0:
                    await self.ws.send_bytes(b"\x01" + opus_bytes)

            # Text / RAG triggers.
            if text_token == self.server.runner.lm_gen.lm_model.rag_token_id:
                self._log.info("[RAG] model emitted RAG token, triggering reference generation")
                assert self._task_group is not None
                await self._send_turn_outputs([("[RET]", "model")])
                await self.stt.flush()
                await self.rag_manager.trigger(
                    task_group=self._task_group,
                    wait_steps=self.server.stt_wait_steps,
                    handle_reference_fn=self._handle_reference_text,
                    context_provider=self.turn_manager.get_context,
                )
            else:
                decoded = self._decode_text_token(text_token)
                await self._send_turn_outputs(self.turn_manager.handle_spoken_text(model_text=decoded))

            self.rag_manager.step()
