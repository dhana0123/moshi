# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Inner Monologue layout from Moshi (arXiv:2410.00037 §3.4.4, Eq. 5–6).

Training data does not use discrete dialogue-state / control tokens. Each 80 ms
frame is a 17-way stack:

    V_s = [W_s, A_s (8 Mimi codebooks), A'_s (8 user codebooks)]

``W`` is the time-aligned transcript of *Moshi's* channel only (PAD / EPAD /
SentencePiece pieces). The user stream has no text channel.

Text delay: a positive ``text_delay_sec`` shifts word timestamps earlier so
text is a prefix to the corresponding acoustic tokens (TTS / dialogue mode).
A negative delay puts audio ahead of text (streaming ASR mode).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    import torch

    from .interleaver import Alignment, Interleaver

MIMI_FRAME_RATE = 12.5
MIMI_FRAME_PERIOD_SEC = 0.08
MIMI_CODEBOOKS = 8
JOINT_STREAMS = 1 + 2 * MIMI_CODEBOOKS  # 17

AGENT_SPEAKER = "SPEAKER_MAIN"
USER_SPEAKER = "SPEAKER_OTHER"

AlignmentTuple = tuple[str, tuple[float, float], str]


def seconds_to_frame(t_sec: float, frame_rate: float = MIMI_FRAME_RATE) -> int:
    """Map a timestamp onto the 12.5 Hz grid (paper t_i, 0-based)."""
    return max(0, int(t_sec * frame_rate))


def apply_text_delay(
    alignments: Sequence[AlignmentTuple],
    text_delay_sec: float,
) -> list[AlignmentTuple]:
    """Shift word times so text leads (positive) or lags (negative) audio."""
    if text_delay_sec == 0:
        return list(alignments)
    out: list[AlignmentTuple] = []
    for word, (start, end), speaker in alignments:
        start_d = start - text_delay_sec
        end_d = end - text_delay_sec
        if end_d <= 0:
            continue
        out.append((word, (max(0.0, start_d), end_d), speaker))
    return out


def agent_alignments_only(
    alignments: Iterable[AlignmentTuple],
    main_speaker: str = AGENT_SPEAKER,
) -> list[AlignmentTuple]:
    """Inner Monologue never transcribes the user channel (paper §3.4.4)."""
    return [a for a in alignments if a[2] == main_speaker]


def build_inner_monologue_stream(
    interleaver: "Interleaver",
    alignments: Sequence[AlignmentTuple] | None,
    duration_sec: float,
    *,
    text_delay_sec: float = 0.0,
    main_speaker: str = AGENT_SPEAKER,
) -> "torch.Tensor":
    """PAD / EPAD / word-piece stream ``W`` of shape ``[1, 1, T]``."""
    delayed = apply_text_delay(list(alignments or []), text_delay_sec)
    delayed = agent_alignments_only(delayed, main_speaker)
    return interleaver.prepare_item(
        delayed if delayed else None,
        duration_sec,
        main_speaker=main_speaker,
    )


def stack_joint_sequence(
    text: "torch.Tensor",
    agent_audio: "torch.Tensor",
    user_audio: "torch.Tensor",
) -> "torch.Tensor":
    """Paper Eq. 6: ``[W, A_1..A_8, A'_1..A'_8]`` → ``[B, 17, T]``.

    ``text`` is ``[B, 1, T]`` or ``[1, T]``. Audio tensors are ``[B, 8, T]``
    or ``[8, T]``. Acoustic delay ``τ`` is applied inside the LM, not here.
    """
    import torch

    if text.dim() == 2:
        text = text.unsqueeze(0)
    if agent_audio.dim() == 2:
        agent_audio = agent_audio.unsqueeze(0)
    if user_audio.dim() == 2:
        user_audio = user_audio.unsqueeze(0)
    if text.shape[-1] != agent_audio.shape[-1] or agent_audio.shape[-1] != user_audio.shape[-1]:
        raise ValueError(
            f"Time mismatch text={tuple(text.shape)} agent={tuple(agent_audio.shape)} "
            f"user={tuple(user_audio.shape)}"
        )
    if agent_audio.shape[1] != MIMI_CODEBOOKS or user_audio.shape[1] != MIMI_CODEBOOKS:
        raise ValueError("Each audio stream must have 8 Mimi codebooks.")
    return torch.cat([text, agent_audio, user_audio], dim=1)


def n_frames_for_duration(duration_sec: float, frame_rate: float = MIMI_FRAME_RATE) -> int:
    return int(math.ceil(duration_sec * frame_rate))
