# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Default Moshi architecture as a freeze/swap research system.

Loading matches inference: CheckpointInfo + get_mimi / get_moshi / tokenizer.
"""

from __future__ import annotations

import typing as tp

import sentencepiece
import torch
from torch import nn

from ..models.compression import MimiModel
from ..models.lm import LMModel, LMOutput
from ..models.loaders import DEFAULT_REPO, CheckpointInfo, get_mimi, get_moshi_lm
from .freeze import apply_freeze, get_component, swap_module

if tp.TYPE_CHECKING:
    from ..conditioners import ConditionTensors


class MoshiSystem(nn.Module):
    """Mimi codec + Moshi LM (Temporal / Helium + Depformer) with named components."""

    def __init__(
        self,
        mimi: MimiModel,
        lm: LMModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor | None = None,
        checkpoint_info: CheckpointInfo | None = None,
        freeze: dict[str, bool] | None = None,
    ):
        super().__init__()
        self.mimi = mimi
        self.lm = lm
        self.text_tokenizer = text_tokenizer
        self.checkpoint_info = checkpoint_info
        if freeze:
            apply_freeze(self, freeze)

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str = DEFAULT_REPO,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        load_weight: bool = True,
        freeze: dict[str, bool] | None = None,
        gradient_checkpointing: bool = True,
        moshi_weights=None,
        mimi_weights=None,
        tokenizer=None,
        config_path=None,
    ) -> "MoshiSystem":
        info = CheckpointInfo.from_hf_repo(
            hf_repo,
            moshi_weights=moshi_weights,
            mimi_weights=mimi_weights,
            tokenizer=tokenizer,
            config_path=config_path,
        )
        mimi = info.get_mimi(device=device)
        mimi.set_num_codebooks(8)
        lm = info.get_moshi(
            device=device,
            dtype=dtype,
            load_weight=load_weight,
            lm_kwargs_overrides={"gradient_checkpointing": gradient_checkpointing},
        )
        spm = info.get_text_tokenizer()
        return cls(mimi, lm, spm, checkpoint_info=info, freeze=freeze)

    @classmethod
    def from_scratch(
        cls,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        freeze: dict[str, bool] | None = None,
        num_codebooks: int = 8,
        **lm_overrides,
    ) -> "MoshiSystem":
        """Random-init default architecture (same constructors as inference)."""
        mimi = get_mimi(None, device=device, num_codebooks=num_codebooks)
        lm = get_moshi_lm(None, device=device, dtype=dtype, lm_kwargs_overrides=lm_overrides)
        return cls(mimi, lm, freeze=freeze)

    @property
    def num_codebooks(self) -> int:
        return self.lm.num_codebooks

    @property
    def dep_q(self) -> int:
        return self.lm.dep_q

    @property
    def n_q(self) -> int:
        return self.lm.n_q

    @property
    def frame_rate(self) -> float:
        return self.mimi.frame_rate

    @property
    def sample_rate(self) -> int:
        return self.mimi.sample_rate

    def component(self, name: str) -> nn.Module:
        return get_component(self, name)

    def freeze_components(self, freeze_map: dict[str, bool]) -> None:
        apply_freeze(self, freeze_map)

    def swap(self, name: str, module: nn.Module) -> nn.Module:
        return swap_module(self, name, module)

    def encode_stereo(self, wav: torch.Tensor) -> torch.Tensor:
        """Encode stereo (or dual-mono) audio to stacked Mimi codes.

        Args:
            wav: [B, 2, T] or [2, T] at 24 kHz. Channel 0 is Moshi, channel 1 is the user.
        Returns:
            codes: [B, 16, S] (8 Moshi + 8 user).
        """
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)
        if wav.dim() != 3 or wav.shape[1] != 2:
            raise ValueError(f"Expected stereo [B, 2, T], got {tuple(wav.shape)}")
        ctx = torch.no_grad() if not any(p.requires_grad for p in self.mimi.parameters()) else torch.enable_grad()
        with ctx:
            moshi_codes = self.mimi.encode(wav[:, :1])
            user_codes = self.mimi.encode(wav[:, 1:2])
        return torch.cat([moshi_codes, user_codes], dim=1)

    def forward(
        self,
        codes: torch.Tensor,
        condition_tensors: tp.Optional["ConditionTensors"] = None,
    ) -> LMOutput:
        """Training forward on interleaved text+audio codes [B, K, T]."""
        return self.lm(codes=codes, condition_tensors=condition_tensors)
