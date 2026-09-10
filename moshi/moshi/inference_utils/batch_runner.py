# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch

from ..models import LMGen, MimiModel

logger = logging.getLogger(__name__)

TSlot = TypeVar("TSlot")

DeliverStep = Callable[..., None]
"""``deliver(slot, *, text_token, pcm_out)`` per row."""


@dataclass
class BatchInput(Generic[TSlot]):
    """Tensors + active slots for one batched step."""

    filtered_pcm_batch: torch.Tensor
    lm_mask_cpu: torch.Tensor
    first_mask_cpu: torch.Tensor
    active: list[tuple[int, TSlot]]
    pcm_batch: torch.Tensor | None = None


class BatchRunner:
    """GPU-only: Mimi encode → LM step(s) → decode."""

    def __init__(
        self,
        mimi: MimiModel,
        lm_gen: LMGen,
        device: str | torch.device | None,
        batch_size: int,
    ) -> None:
        self.mimi = mimi
        self.lm_gen = lm_gen
        self.device = device
        self.batch_size = batch_size
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)

        self.mimi.streaming_forever(batch_size)
        self.lm_gen.streaming_forever(batch_size)

    def warmup(self) -> None:
        """Compile CUDA kernels for full batch; reset streaming state (no reference LLM)."""
        full_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        for _ in range(4):
            chunk = torch.zeros(
                self.batch_size,
                1,
                self.frame_size,
                dtype=torch.float32,
                device=self.device,
            )
            self.mimi.set_exec_mask(full_mask)
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                lm_codes = codes[:, : self.lm_gen.needed_tokens, c : c + 1].to(self.device)
                self.lm_gen.set_exec_mask(full_mask)
                tokens = self.lm_gen.step(lm_codes)
                if tokens is None:
                    continue
                self.mimi.set_exec_mask(full_mask)
                _ = self.mimi.decode(tokens[:, 1:].clamp(min=0).to(self.device))
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def run_step(self, g: BatchInput[Any], deliver: DeliverStep) -> bool:
        """One encode→LM→decode iteration; ``deliver`` pushes each row to the slot object."""
        be = time.time()
        first_mask_dev = g.first_mask_cpu.to(self.device)
        has_first = g.first_mask_cpu.any()
        if has_first:
            self.mimi.reset_streaming(reset_mask=first_mask_dev)
            self.lm_gen.reset_streaming(reset_mask=first_mask_dev)

        self.mimi.set_exec_mask(g.lm_mask_cpu.to(self.device))
        codes = self.mimi.encode(g.filtered_pcm_batch)

        lm_mask_dev = g.lm_mask_cpu.to(self.device)
        ungenerated = self.lm_gen.lm_model.ungenerated_token_id
        assert codes.shape[-1] == 1, codes.shape
        for c in range(codes.shape[-1]):
            lm_codes = codes[:, : self.lm_gen.needed_tokens, c : c + 1].to(self.device)
            self.lm_gen.set_exec_mask(lm_mask_dev)
            self.lm_gen.apply_pending_streaming_sum_condition(g.lm_mask_cpu)
            tokens = self.lm_gen.step(lm_codes)
            if tokens is None:
                continue
            assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1

            audio_tokens = tokens[:, 1:]
            decode_mask_cpu = (audio_tokens != ungenerated).all(dim=1).squeeze(-1).cpu()
            decode_mask_dev = decode_mask_cpu.to(self.device)

            self.mimi.set_exec_mask(decode_mask_dev)
            audio_pcm = self.mimi.decode(audio_tokens.clamp(min=0).to(self.device)).cpu()

            for b, slot in g.active:
                text_token = int(tokens[b, 0, 0].item())
                pcm_out = audio_pcm[b, 0] if decode_mask_cpu[b] else None
                deliver(
                    slot,
                    text_token=text_token,
                    pcm_out=pcm_out,
                )

        elapsed_ms = 1000 * (time.time() - be)
        if elapsed_ms >= 77:
            logger.warning("batched step (%d/%d active) took %.1fms", len(g.active), self.batch_size, elapsed_ms)
        return True
