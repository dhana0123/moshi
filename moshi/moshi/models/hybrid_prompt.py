# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Hybrid System Prompt: voice clip + role text, with sine user and silence delimiters.

Matches the PersonaPlex prefix layout:

* Voice: agent audio = clip, agent text = PAD, user audio = 440 Hz sine codes
* Silence delimiter (~0.5 s)
* Role text: agent text = ``<system> ... <system>`` tokens, agent audio = silence, user = sine
* Silence delimiter
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

# Hardcoded Mimi RVQ codes (8 codebooks) from PersonaPlex.
SILENCE_TOKENS = np.array([948, 243, 1178, 546, 1736, 1030, 1978, 2008], dtype=np.int64)
SINE_TOKENS = np.array([430, 1268, 381, 1611, 1095, 1495, 56, 472], dtype=np.int64)

DEFAULT_SYSTEM_PROMPT = "You enjoy having a good conversation."
DEFAULT_SILENCE_SECONDS = 0.5
FRAME_RATE_HZ = 12.5


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as the model expects if they are missing."""
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def silence_frame_count(frame_rate: float = FRAME_RATE_HZ, seconds: float = DEFAULT_SILENCE_SECONDS) -> int:
    return max(1, int(seconds * frame_rate))


def _codes_1d(src: np.ndarray, n: int) -> torch.Tensor:
    if n <= len(src):
        vals = src[:n]
    else:
        vals = np.resize(src, n)
    return torch.as_tensor(vals, dtype=torch.long)


def _block(
    text: torch.Tensor,
    agent: torch.Tensor,
    user: torch.Tensor,
) -> torch.Tensor:
    """Stack [1, 1+n_q, T] from [1, T] text and [dep_q, T] / [user_q, T] audio."""
    return torch.cat([text.unsqueeze(0), agent, user], dim=0).unsqueeze(0)


def build_hybrid_system_prefix(
    text_token_ids: list[int] | None,
    *,
    n_q: int,
    dep_q: int,
    pad_id: int = 3,
    silence_frames: int = 6,
    voice_codes: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build prefix codes ``[1, 1 + n_q, T]``.

    ``voice_codes`` is agent audio ``[dep_q, T_v]`` or ``[1, dep_q, T_v]``.
    """
    user_q = n_q - dep_q
    if user_q < 0:
        raise ValueError(f"n_q={n_q} must be >= dep_q={dep_q}")

    def pad_text(t: int) -> torch.Tensor:
        return torch.full((1, t), pad_id, dtype=torch.long, device=device)

    def sil(t: int) -> torch.Tensor:
        row = _codes_1d(SILENCE_TOKENS, dep_q).to(device=device)
        return row.view(dep_q, 1).expand(dep_q, t).clone()

    def sine(t: int) -> torch.Tensor:
        if user_q == 0:
            return torch.zeros(0, t, dtype=torch.long, device=device)
        row = _codes_1d(SINE_TOKENS, user_q).to(device=device)
        return row.view(user_q, 1).expand(user_q, t).clone()

    chunks: list[torch.Tensor] = []

    if voice_codes is not None:
        vc = voice_codes
        if vc.dim() == 3:
            vc = vc[0]
        if vc.dim() != 2:
            raise ValueError(f"voice_codes should be [dep_q, T], got {tuple(voice_codes.shape)}")
        if vc.shape[0] != dep_q:
            if vc.shape[0] > dep_q:
                vc = vc[:dep_q]
            else:
                vc = torch.nn.functional.pad(vc, (0, 0, 0, dep_q - vc.shape[0]))
        tv = vc.shape[1]
        if tv > 0:
            chunks.append(_block(pad_text(tv), vc.to(device=device, dtype=torch.long), sine(tv)))
            chunks.append(_block(pad_text(silence_frames), sil(silence_frames), sine(silence_frames)))

    ids = list(text_token_ids or [])
    if ids:
        tt = len(ids)
        text = torch.tensor(ids, dtype=torch.long, device=device).view(1, tt)
        chunks.append(_block(text, sil(tt), sine(tt)))
        chunks.append(_block(pad_text(silence_frames), sil(silence_frames), sine(silence_frames)))

    if not chunks:
        return torch.zeros(1, 1 + n_q, 0, dtype=torch.long, device=device)
    return torch.cat(chunks, dim=-1)


def apply_prefix_loss_mask(
    text_mask: torch.Tensor,
    audio_mask: torch.Tensor,
    prefix_frames: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero loss on the hybrid prefix. ``prefix_frames`` is ``[B]``."""
    t = text_mask.shape[-1]
    time = torch.arange(t, device=text_mask.device)
    keep = time[None, :] >= prefix_frames.to(text_mask.device).view(-1, 1)
    text_keep = keep[:, None, :]
    audio_keep = keep[:, None, :]
    return text_mask & text_keep, audio_mask & audio_keep.expand_as(audio_mask)


def encode_wav_agent_codes(mimi, wav: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Encode mono wav to agent codes ``[K, T]`` (non-streaming)."""
    device = next(mimi.parameters()).device
    audio = torch.as_tensor(wav, device=device, dtype=torch.float32)
    if audio.dim() == 1:
        audio = audio.view(1, 1, -1)
    elif audio.dim() == 2:
        audio = audio[:1].unsqueeze(0)
    elif audio.dim() == 3:
        audio = audio[:1, :1]
    else:
        raise ValueError(f"Expected wav rank 1-3, got {tuple(audio.shape)}")
    with torch.no_grad():
        codes = mimi.encode(audio)
    if codes.dim() == 3:
        codes = codes[0]
    return codes


def iter_code_frames(codes: torch.Tensor) -> Iterator[torch.Tensor]:
    """Yield ``[1, K, 1]`` frames from ``[K, T]`` or ``[1, K, T]``."""
    if codes.dim() == 2:
        codes = codes.unsqueeze(0)
    t = codes.shape[-1]
    for i in range(t):
        yield codes[:, :, i : i + 1]


@dataclass
class HybridPromptConfig:
    enabled: bool = False
    silence_seconds: float = DEFAULT_SILENCE_SECONDS
    voice_prompt_dir: str | None = None
    system_prompts: list[str] = field(default_factory=lambda: [DEFAULT_SYSTEM_PROMPT])
    proba: float = 1.0
    pad_id: int = 3

    def silence_frames(self, frame_rate: float) -> int:
        return silence_frame_count(frame_rate, self.silence_seconds)

    def sample_system_prompt(self, rng: np.random.Generator) -> str:
        prompts = self.system_prompts or [DEFAULT_SYSTEM_PROMPT]
        return str(prompts[int(rng.integers(0, len(prompts)))])

    def list_voice_files(self) -> list[Path]:
        if not self.voice_prompt_dir:
            return []
        root = Path(self.voice_prompt_dir)
        if not root.is_dir():
            return []
        files = sorted(
            p for p in root.iterdir() if p.suffix.lower() in {".wav", ".flac", ".mp3", ".pt"}
        )
        return files
