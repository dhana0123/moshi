# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Paper Eq. 7: text CE plus weighted audio CE (semantic α=100, acoustic α=1)."""

import torch
from torch.nn import functional as F


def compute_loss_with_mask(
    logits: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    mode: str,
    first_codebook_weight_multiplier: float = 100.0,
    text_padding_weight: float = 0.5,
    text_padding_ids: set[int] | None = None,
) -> torch.Tensor:
    target = torch.where(target_mask, target, torch.zeros_like(target))
    weights = target_mask.float()
    if mode == "audio":
        weights[:, 0] *= first_codebook_weight_multiplier
    elif mode == "text":
        assert text_padding_ids is not None
        for token_id in text_padding_ids:
            weights[target == token_id] *= text_padding_weight
    else:
        raise ValueError(f"Unknown loss mode {mode!r}")

    logits_flat = logits.reshape(-1, logits.size(-1)).float()
    target_flat = target.reshape(-1)
    weights_flat = weights.reshape(-1)
    token_loss = F.cross_entropy(logits_flat, target_flat, reduction="none")
    token_loss = torch.where(weights_flat > 0.0, token_loss * weights_flat, torch.zeros_like(token_loss))
    denom = torch.sum(weights_flat).clamp_min(1e-8)
    return torch.sum(token_loss) / denom


def moshi_loss(
    text_logits: torch.Tensor,
    text_target: torch.Tensor,
    text_mask: torch.Tensor,
    audio_logits: torch.Tensor,
    audio_target: torch.Tensor,
    audio_mask: torch.Tensor,
    text_padding_ids: set[int],
    first_codebook_weight_multiplier: float = 100.0,
    text_padding_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    text_loss = compute_loss_with_mask(
        text_logits,
        text_target,
        text_mask,
        mode="text",
        text_padding_weight=text_padding_weight,
        text_padding_ids=text_padding_ids,
    )
    audio_loss = compute_loss_with_mask(
        audio_logits,
        audio_target,
        audio_mask,
        mode="audio",
        first_codebook_weight_multiplier=first_codebook_weight_multiplier,
    )
    return text_loss + audio_loss, text_loss, audio_loss
