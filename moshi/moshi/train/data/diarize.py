# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Two-speaker diarization → dual-channel gating for Moshi train packs.

IndicVoices conversations are usually mono mixes. We run pyannote with
``num_speakers=2``, then time-gate the mix onto left (agent) / right (user).
Agent = speaker with more speaking duration. Clips where either speaker has
less than ``MIN_SPEAKER_SEC`` of speech are skipped.
"""

from __future__ import annotations

import logging
import os
import tempfile
import wave
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("moshi.prepare.diarize")

MIN_SPEAKER_SEC = 0.5
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

Turn = tuple[str, float, float]  # speaker_id, start_sec, end_sec

_pipeline_cache: dict[str, object] = {}


class DiarizationSkip(Exception):
    """Clip is not usable as a 2-speaker duplex example."""


@dataclass
class DiarizationResult:
    turns: list[Turn]
    agent_id: str
    user_id: str
    durations: dict[str, float]


def _write_temp_wav(mono: np.ndarray, sample_rate: int) -> str:
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    pcm = np.clip(np.asarray(mono, dtype=np.float32), -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return path


def load_diarization_pipeline(device: str = "cuda"):
    """Load and cache pyannote speaker-diarization-3.1 (needs HF_TOKEN + model accept)."""
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise ImportError(
            "Install pyannote.audio (`pip install 'pyannote.audio>=3.1,<4'` or moshi[data]) "
            "for --diarize."
        ) from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        logger.warning(
            "HF_TOKEN not set — gated pyannote models will fail. "
            "export HF_TOKEN=... after accepting model cards."
        )
    key = f"{DIARIZATION_MODEL}|{device}|{bool(token)}"
    if key in _pipeline_cache:
        return _pipeline_cache[key]

    gate_help = (
        "Accept these gated repos (same HF account as HF_TOKEN), then retry:\n"
        "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
        "  https://huggingface.co/pyannote/segmentation-3.0\n"
        "If pyannote.audio>=4 is installed it may also need:\n"
        "  https://huggingface.co/pyannote/speaker-diarization-community-1\n"
        "Preferred fix: pip install 'pyannote.audio>=3.1,<4'"
    )

    kwargs = {"token": token} if token else {}
    try:
        logger.info("Loading diarization pipeline %s …", DIARIZATION_MODEL)
        try:
            pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, **kwargs)
        except TypeError:
            # Older pyannote used use_auth_token=
            pipeline = Pipeline.from_pretrained(
                DIARIZATION_MODEL, use_auth_token=token or True
            )
    except Exception as exc:
        raise RuntimeError(f"Failed to load {DIARIZATION_MODEL}.\n{gate_help}") from exc
    if pipeline is None:
        raise RuntimeError(f"Failed to load {DIARIZATION_MODEL}.\n{gate_help}")
    if device.startswith("cuda") and torch.cuda.is_available():
        pipeline.to(torch.device(device))
    else:
        pipeline.to(torch.device("cpu"))
    _pipeline_cache[key] = pipeline
    logger.info("Diarization ready: %s", DIARIZATION_MODEL)
    return pipeline


def _annotation_to_turns(annotation) -> list[Turn]:
    turns: list[Turn] = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        start = float(segment.start)
        end = float(segment.end)
        if end <= start:
            continue
        turns.append((str(speaker), start, end))
    return turns


def speaker_durations(turns: list[Turn]) -> dict[str, float]:
    dur: dict[str, float] = defaultdict(float)
    for spk, start, end in turns:
        dur[spk] += end - start
    return dict(dur)


def pick_agent_user(durations: dict[str, float]) -> tuple[str, str]:
    if len(durations) < 2:
        raise DiarizationSkip(f"Need 2 speakers, got {list(durations)}")
    ranked = sorted(durations.items(), key=lambda kv: (-kv[1], kv[0]))
    agent_id, user_id = ranked[0][0], ranked[1][0]
    return agent_id, user_id


def validate_two_speakers(
    durations: dict[str, float],
    *,
    min_sec: float = MIN_SPEAKER_SEC,
) -> None:
    if len(durations) != 2:
        raise DiarizationSkip(f"Expected exactly 2 speakers, got {durations}")
    for spk, d in durations.items():
        if d < min_sec:
            raise DiarizationSkip(
                f"Speaker {spk} only {d:.2f}s (< {min_sec}s); skip clip"
            )


def diarize_two_speakers(
    mono: np.ndarray,
    sample_rate: int,
    *,
    device: str = "cuda",
    min_sec: float = MIN_SPEAKER_SEC,
) -> DiarizationResult:
    """Run pyannote with num_speakers=2; raise DiarizationSkip if not usable."""
    mono = np.asarray(mono, dtype=np.float32)
    if mono.ndim > 1:
        mono = mono.mean(axis=0) if mono.shape[0] < mono.shape[-1] else mono.mean(axis=-1)

    pipeline = load_diarization_pipeline(device=device)
    path = _write_temp_wav(mono, sample_rate)
    try:
        output = pipeline(path, num_speakers=2)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    # pyannote 3.x may return Annotation or DiarizeOutput with .speaker_diarization
    if hasattr(output, "speaker_diarization"):
        annotation = output.speaker_diarization
    else:
        annotation = output

    turns = _annotation_to_turns(annotation)
    if not turns:
        raise DiarizationSkip("Empty diarization")
    durations = speaker_durations(turns)
    # With num_speakers=2 we may still get 1 if clustering collapses; enforce 2.
    if len(durations) < 2:
        raise DiarizationSkip(f"Diarization found {len(durations)} speaker(s): {durations}")
    # Keep top-2 by duration if somehow >2
    if len(durations) > 2:
        top = sorted(durations.items(), key=lambda kv: -kv[1])[:2]
        keep = {t[0] for t in top}
        turns = [t for t in turns if t[0] in keep]
        durations = speaker_durations(turns)
    validate_two_speakers(durations, min_sec=min_sec)
    agent_id, user_id = pick_agent_user(durations)
    return DiarizationResult(
        turns=turns, agent_id=agent_id, user_id=user_id, durations=durations
    )


def mono_to_stereo_from_diarization(
    mono: np.ndarray,
    sample_rate: int,
    result: DiarizationResult,
) -> np.ndarray:
    """Gate mono mix into [2, T]: agent=left, user=right. Overlap → both channels."""
    mono = np.asarray(mono, dtype=np.float32)
    if mono.ndim > 1:
        mono = mono.mean(axis=0) if mono.shape[0] < mono.shape[-1] else mono.mean(axis=-1)
    n = mono.shape[-1]
    agent = np.zeros(n, dtype=np.float32)
    user = np.zeros(n, dtype=np.float32)
    for spk, start, end in result.turns:
        i0 = max(0, int(round(start * sample_rate)))
        i1 = min(n, int(round(end * sample_rate)))
        if i1 <= i0:
            continue
        if spk == result.agent_id:
            agent[i0:i1] = mono[i0:i1]
        elif spk == result.user_id:
            user[i0:i1] = mono[i0:i1]
    return np.stack([agent, user], axis=0)


def diarize_mono_to_stereo(
    mono: np.ndarray,
    sample_rate: int,
    *,
    device: str = "cuda",
    min_sec: float = MIN_SPEAKER_SEC,
) -> np.ndarray:
    """Full path: diarize → gate → [2, T]. Raises DiarizationSkip on failure."""
    result = diarize_two_speakers(mono, sample_rate, device=device, min_sec=min_sec)
    return mono_to_stereo_from_diarization(mono, sample_rate, result)
