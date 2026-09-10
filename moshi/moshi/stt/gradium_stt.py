# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import os
from logging import getLogger
from typing import Callable

import gradium
import numpy as np

from .stt import STTWordMessage


logger = getLogger(__name__)

SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 1920


class GradiumSpeechToText:
    def __init__(
        self,
        expected_language: str | None,
        vad_callback: Callable[[float], None] | None = None,
        use_flush: bool = True,
    ):
        self.client = gradium.client.GradiumClient(api_key=os.environ["STT_API_KEY"])
        self.stt = None
        self.expected_language = expected_language
        self.vad_callback = vad_callback
        self.use_flush = use_flush
        self._flushing = False
        self._flush_id = 1
        self.audio_queue: asyncio.Queue[np.ndarray | None | str] = asyncio.Queue()
        self.flush_queue: asyncio.Queue[None] = asyncio.Queue()

    async def send_audio(self, audio: np.ndarray) -> None:
        if audio.ndim != 1:
            raise ValueError(f"Expected 1D array, got {audio.shape=}")

        if audio.dtype != np.float32:
            raise ValueError(f"Expected float32 array, got {audio.dtype=}")

        await self.audio_queue.put(audio)

    async def flush(self):
        if self.use_flush:
            await self.audio_queue.put("flush")
            await self.flush_queue.get()

    async def __aiter__(self):
        assert self.stt is not None
        json_config = {"language": "en", "delay_in_frames": 16}

        async def push_audio(stt):
            logger.info("STT audio feed started.")
            while True:
                audio = await self.audio_queue.get()
                if audio is None:
                    logger.info("STT audio feed exited.")
                    break
                elif isinstance(audio, str):
                    assert audio == "flush"
                    logger.info("Flushing.")
                    await stt.send_flush(flush_id=self._flush_id)
                    self._flush_id += 1
                else:
                    await stt.send_audio(audio)

        async with self.client.stt_realtime(model_name="default", input_format="pcm", json_config=json_config) as stt:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(push_audio(stt))
                logger.info("STT started.")
                try:
                    while True:
                        async for msg in stt:
                            if msg["type"] == "text":
                                yield STTWordMessage(text=msg["text"] + " ", start_time=0.0, type="Word")
                            elif msg["type"] == "flushed":
                                logger.info("Flush is over.")
                                await self.flush_queue.put(None)
                            elif msg["type"] == "step":
                                if self.vad_callback:
                                    end_of_turn_proba = msg["vad"][2]["inactivity_prob"]
                                    # Gradium provides inactivity probability (high = silent).
                                    # TurnManager expects low score for user speech and high score for silence.
                                    self.vad_callback(end_of_turn_proba)
                finally:
                    await self.audio_queue.put(None)
                    await self.flush_queue.put(None)
                    logger.info("STT exited.")

    async def start_up(self):
        json_config = {"language": "en", "delay_in_frames": 16}
        self.stt = self.client.stt_realtime(model_name="default", input_format="pcm", json_config=json_config)
        # A bit ugly to not used context manager here, but easier to match the existing code
        # taken from Unmute.
        await self.stt.__aenter__()

    async def shutdown(self):
        assert self.stt is not None
        await self.stt.__aexit__(None, None, None)
