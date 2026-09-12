# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Build Moshi Inner Monologue training data (no dialogue control tokens).

Writes the on-disk format consumed by ``moshi.train``:

* 24 kHz stereo WAV (left = agent / Moshi, right = user), same ``t=0`` and length
* sibling ``.json`` with word alignments on the **agent** channel only
* ``jsonl`` rows ``{"path", "duration"}``

Mimi RVQ (8+8 codebooks at 12.5 Hz) and PAD/EPAD interleaving happen in the
train loader. Pass ``--encode-mimi`` to also dump the 17-stream matrix for
inspection.
"""

from __future__ import annotations

import argparse
import json
import logging
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .inner_monologue import AGENT_SPEAKER, MIMI_FRAME_RATE

logger = logging.getLogger("moshi.prepare")

SAMPLE_RATE = 24_000
INDICVOICES_FOCUS = ("hindi", "telugu", "kannada", "tamil")
INDICVOICES_CONFIG_TO_LANG = {
    "hindi": "hi",
    "telugu": "te",
    "kannada": "kn",
    "tamil": "ta",
}
CONVERSATION_MARKERS = (
    "conversation",
    "conversational",
    "roleplay",
    "role-play",
    "role_play",
    "spontaneous",
)


@dataclass
class PackedClip:
    wav_path: Path
    json_path: Path
    duration: float
    n_agent_words: int


def write_pcm16_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write float audio in ``[-1, 1]`` as PCM16. ``audio`` is ``[C, T]`` or ``[T]``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        x = x[None]
    if x.ndim != 2:
        raise ValueError(f"Expected [C, T], got {x.shape}")
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    n_ch, n_samp = pcm.shape
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_ch)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.T.tobytes())


def read_pcm16_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        n_ch = wf.getnchannels()
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    pcm = np.frombuffer(raw, dtype=np.int16).reshape(-1, n_ch).T
    return pcm.astype(np.float32) / 32767.0, sr


def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x.astype(np.float32, copy=False)
    n_src = x.shape[-1]
    n_dst = int(round(n_src * dst_sr / src_sr))
    if n_dst <= 1:
        return np.zeros((*x.shape[:-1], max(n_dst, 0)), dtype=np.float32)
    src_t = np.linspace(0.0, 1.0, n_src, endpoint=False)
    dst_t = np.linspace(0.0, 1.0, n_dst, endpoint=False)
    if x.ndim == 1:
        return np.interp(dst_t, src_t, x).astype(np.float32)
    return np.stack([np.interp(dst_t, src_t, ch) for ch in x]).astype(np.float32)


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    if x.shape[0] == 1:
        return x[0]
    return x.mean(axis=0)


def pack_stereo(
    agent: np.ndarray,
    user: np.ndarray,
    sample_rate: int,
    *,
    target_sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Time-sync two channels to identical length at 1.0× (paper multi-stream)."""
    a = resample_linear(_to_mono(np.asarray(agent, dtype=np.float32)), sample_rate, target_sr)
    u = resample_linear(_to_mono(np.asarray(user, dtype=np.float32)), sample_rate, target_sr)
    t = max(a.shape[-1], u.shape[-1])
    if a.shape[-1] < t:
        a = np.pad(a, (0, t - a.shape[-1]))
    if u.shape[-1] < t:
        u = np.pad(u, (0, t - u.shape[-1]))
    peak = max(float(np.max(np.abs(a))), float(np.max(np.abs(u))), 1e-6)
    if peak > 1.0:
        a = a / peak
        u = u / peak
    return np.stack([a, u], axis=0)


def alignments_payload(
    words: list[tuple[str, tuple[float, float]]],
    *,
    speaker: str = AGENT_SPEAKER,
    text_delay_sec: float = 0.0,
    provenance: dict | None = None,
) -> dict:
    """Sidecar JSON with acoustic word times + ASR/timer model labels."""
    payload = {
        "alignments": [
            [w, [float(s), float(e)], speaker] for w, (s, e) in words
        ],
        "inner_monologue": {
            "speaker": speaker,
            "frame_rate_hz": MIMI_FRAME_RATE,
            "recommended_audio_delay_sec": text_delay_sec,
            "user_text": False,
            "control_tokens": False,
        },
    }
    if provenance:
        payload["extraction"] = provenance
    return payload


def write_sidecar(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(
    jsonl_path: Path,
    wav_path: Path,
    duration: float,
    *,
    provenance: dict | None = None,
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    rec: dict = {"path": str(wav_path.resolve()), "duration": float(duration)}
    if provenance:
        rec["extraction"] = provenance
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def is_conversation_row(row: dict) -> bool:
    blob = f"{row.get('task_name', '')} {row.get('scenario', '')}".lower()
    return any(m in blob for m in CONVERSATION_MARKERS)


def parse_indicvoices_configs(raw: str | None) -> list[str]:
    """Comma-separated HF config names; default = hindi,telugu,kannada,tamil."""
    if not raw or not str(raw).strip():
        return list(INDICVOICES_FOCUS)
    configs = [c.strip().lower() for c in str(raw).split(",") if c.strip()]
    return configs or list(INDICVOICES_FOCUS)


def asr_lang_for_indicvoices_config(config: str, language_override: str | None) -> str:
    if language_override:
        return language_override
    return INDICVOICES_CONFIG_TO_LANG.get(config.lower(), "hi")


def load_alignments_json(path: Path) -> list[tuple[str, tuple[float, float]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "alignments" in data:
        rows = data["alignments"]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"No alignments in {path}")
    out = []
    for row in rows:
        word, ts, *_rest = row
        out.append((str(word), (float(ts[0]), float(ts[1]))))
    return out


def save_clip(
    out_dir: Path,
    stem: str,
    stereo: np.ndarray,
    words: list[tuple[str, tuple[float, float]]],
    jsonl_path: Path,
    *,
    text_delay_sec: float,
    sample_rate: int = SAMPLE_RATE,
    provenance: dict | None = None,
) -> PackedClip:
    wav_path = out_dir / "wav" / f"{stem}.wav"
    json_path = out_dir / "wav" / f"{stem}.json"
    write_pcm16_wav(wav_path, stereo, sample_rate)
    duration = stereo.shape[-1] / sample_rate
    write_sidecar(
        json_path,
        alignments_payload(words, text_delay_sec=text_delay_sec, provenance=provenance),
    )
    append_jsonl(jsonl_path, wav_path, duration, provenance=provenance)
    return PackedClip(wav_path, json_path, duration, len(words))


def iter_stereo_dir(root: Path) -> Iterator[tuple[str, np.ndarray, int]]:
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".wav", ".flac", ".mp3", ".ogg"}:
            continue
        try:
            import sphn

            wav, sr = sphn.read(str(path))
            yield path.stem, np.asarray(wav, dtype=np.float32), int(sr)
        except Exception:
            if path.suffix.lower() != ".wav":
                logger.warning("Skip (need sphn for %s): %s", path.suffix, path)
                continue
            wav, sr = read_pcm16_wav(path)
            yield path.stem, wav, sr


def process_stereo_array(
    name: str,
    wav: np.ndarray,
    sr: int,
    *,
    language: str,
    asr_backend: str,
    asr_model: str | None,
    asr_device: str,
    alignments: list[tuple[str, tuple[float, float]]] | None,
    agent_channel: int,
    user_channel: int,
    allow_silent_user: bool,
    diarize: bool,
    text_delay_sec: float,
    out_dir: Path,
    jsonl_path: Path,
) -> PackedClip:
    from .asr import transcribe_agent_words
    from .diarize import DiarizationSkip, diarize_mono_to_stereo

    x = np.asarray(wav, dtype=np.float32)
    if x.ndim == 2 and x.shape[0] not in (1, 2) and x.shape[1] in (1, 2):
        x = x.T
    if x.ndim == 1:
        x = x[None]
    if x.shape[0] == 1:
        mono = x[0]
        if diarize:
            try:
                stereo = diarize_mono_to_stereo(mono, sr, device=asr_device)
                stereo = pack_stereo(stereo[0], stereo[1], sr)
            except DiarizationSkip as exc:
                raise DiarizationSkip(f"{name}: {exc}") from exc
        elif allow_silent_user:
            stereo = pack_stereo(mono, np.zeros_like(mono), sr)
        else:
            raise ValueError(
                f"{name}: mono file. Use --diarize (default) for 2-speaker gating, "
                "or --allow-silent-user / provide stereo."
            )
    else:
        if agent_channel >= x.shape[0] or user_channel >= x.shape[0]:
            raise ValueError(f"{name}: not enough channels ({x.shape[0]})")
        stereo = pack_stereo(x[agent_channel], x[user_channel], sr)
    agent = stereo[0]
    provenance: dict | None = None
    if alignments is None:
        result = transcribe_agent_words(
            agent,
            SAMPLE_RATE,
            language=language,
            backend=asr_backend,
            asr_model=asr_model,
            device=asr_device,
        )
        words = result.words
        provenance = result.provenance_dict()
    else:
        words = alignments
        provenance = {
            "transcript_backend": "alignments-json",
            "transcript_model": "external",
            "timer_backend": "alignments-json",
            "timer_model": "external",
            "language": language,
        }
    return save_clip(
        out_dir,
        name,
        stereo,
        words,
        jsonl_path,
        text_delay_sec=text_delay_sec,
        provenance=provenance,
    )


def _audio_column_name(features) -> str:
    names = list(getattr(features, "keys", lambda: features)())
    for key in ("audio_filepath", "audio"):
        if key in names:
            return key
    raise KeyError(f"No audio column in features: {names}")


def decode_hf_audio(audio) -> tuple[np.ndarray, int]:
    """Decode HF Audio path/bytes/array without torchcodec (soundfile)."""
    if audio is None:
        raise ValueError("audio is None")
    # Legacy / already-decoded
    if isinstance(audio, dict) and audio.get("array") is not None:
        return np.asarray(audio["array"], dtype=np.float32), int(
            audio.get("sampling_rate") or SAMPLE_RATE
        )
    # path/bytes dict from Audio(decode=False)
    if isinstance(audio, dict):
        raw = audio.get("bytes")
        path = audio.get("path")
        try:
            import soundfile as sf
        except ImportError as exc:
            raise ImportError("pip install soundfile to decode IndicVoices audio") from exc
        import io

        if raw is not None:
            arr, sr = sf.read(io.BytesIO(raw), always_2d=False, dtype="float32")
        elif path:
            local = Path(path)
            if local.exists():
                arr, sr = sf.read(str(local), always_2d=False, dtype="float32")
            else:
                # HF streaming / zip / hub path — open via datasets xopen
                try:
                    from datasets.utils.file_utils import xopen
                except ImportError as exc:
                    raise ImportError(
                        "Need datasets to open remote audio paths, or pip install torchcodec"
                    ) from exc
                with xopen(path, "rb") as f:
                    blob = f.read()
                arr, sr = sf.read(io.BytesIO(blob), always_2d=False, dtype="float32")
        else:
            raise ValueError(f"Audio dict has neither array, bytes, nor path: {audio.keys()}")
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            # soundfile returns [T, C]; packer expects [C, T] or mono
            arr = arr.T
        return arr, int(sr)
    raise TypeError(f"Unsupported audio payload type: {type(audio)}")


def process_indicvoices(
    *,
    language_config: str,
    split: str,
    max_clips: int | None,
    streaming: bool,
    **kwargs,
) -> list[PackedClip]:
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise ImportError("pip install datasets huggingface_hub to use --indicvoices") from exc

    ds = load_dataset("ai4bharat/IndicVoices", language_config, split=split, streaming=streaming)
    # datasets>=4 defaults to torchcodec; we decode with soundfile instead.
    audio_col = _audio_column_name(ds.features)
    ds = ds.cast_column(audio_col, Audio(decode=False))
    if streaming and hasattr(ds, "decode"):
        ds = ds.decode(False)
    clips: list[PackedClip] = []
    for i, row in enumerate(ds):
        if not is_conversation_row(row):
            continue
        audio = row.get(audio_col) or row.get("audio_filepath") or row.get("audio")
        if audio is None:
            continue
        try:
            arr, sr = decode_hf_audio(audio)
        except Exception as err:
            logger.warning("Skip IndicVoices audio row %s: %s", i, err)
            continue
        stem = f"{language_config}_{split}_{i:08d}"
        try:
            clips.append(process_stereo_array(stem, arr, sr, alignments=None, **kwargs))
        except Exception as err:
            from .asr import AlignmentError
            from .diarize import DiarizationSkip

            if isinstance(err, (DiarizationSkip, AlignmentError)):
                logger.warning("Skip %s: %s", stem, err)
            else:
                logger.exception("Failed IndicVoices row %s", stem)
            continue
        if max_clips is not None and len(clips) >= max_clips:
            break
    return clips


def process_indicvoices_many(
    *,
    language_configs: list[str],
    split: str,
    max_clips_per_lang: int | None,
    streaming: bool,
    language_override: str | None,
    common: dict,
) -> list[PackedClip]:
    """Pack conversational rows for each IndicVoices config (default: 4 focus langs)."""
    all_clips: list[PackedClip] = []
    for config in language_configs:
        lang = asr_lang_for_indicvoices_config(config, language_override)
        kwargs = {**common, "language": lang}
        logger.info(
            "IndicVoices config=%s asr_lang=%s max_clips_per_lang=%s",
            config,
            lang,
            max_clips_per_lang,
        )
        part = process_indicvoices(
            language_config=config,
            split=split,
            max_clips=max_clips_per_lang,
            streaming=streaming,
            **kwargs,
        )
        logger.info("Packed %d conversation rows from %s", len(part), config)
        all_clips.extend(part)
    return all_clips


def make_dummy_dialogue(
    duration_sec: float = 2.56,
    sr: int = SAMPLE_RATE,
) -> tuple[np.ndarray, list[tuple[str, tuple[float, float]]]]:
    """Deterministic dual-channel clip for pipeline tests (agent then overlap)."""
    del duration_sec  # first sample is fixed-length overlap
    _stem, stereo, words, _desc = make_sample_dialogues(sr=sr)[0]
    return stereo, words


def make_sample_dialogues(
    sr: int = SAMPLE_RATE,
) -> list[tuple[str, np.ndarray, list[tuple[str, tuple[float, float]]], str]]:
    """Three short inspectable clips: overlap, pause, agent-only turn.

    Each item: ``(stem, stereo[C,T], words, description)``.
    """

    def _tone(freq: float, t: np.ndarray, gate: np.ndarray) -> np.ndarray:
        return (0.1 * np.sin(2 * np.pi * freq * t) * gate).astype(np.float32)

    samples: list[tuple[str, np.ndarray, list[tuple[str, tuple[float, float]]], str]] = []

    # 1) Overlap: agent then user barge-in
    dur = 2.56
    t = np.arange(int(dur * sr), dtype=np.float32) / sr
    agent = _tone(220.0, t, ((t >= 0.2) & (t < 1.2)).astype(np.float32))
    user = _tone(330.0, t, ((t >= 0.9) & (t < 2.0)).astype(np.float32))
    samples.append(
        (
            "01_overlap",
            np.stack([agent, user], axis=0),
            [("hello", (0.20, 0.55)), ("there", (0.58, 1.10))],
            "Agent speaks, user overlaps mid-turn (no control tokens).",
        )
    )

    # 2) Pause: agent, silence, agent continues; user quiet
    dur = 3.2
    t = np.arange(int(dur * sr), dtype=np.float32) / sr
    agent = _tone(
        240.0,
        t,
        (((t >= 0.15) & (t < 0.9)) | ((t >= 2.0) & (t < 2.8))).astype(np.float32),
    )
    user = np.zeros_like(t)
    samples.append(
        (
            "02_pause",
            np.stack([agent, user], axis=0),
            [
                ("namaste", (0.15, 0.55)),
                ("ji", (0.58, 0.85)),
                ("kaise", (2.00, 2.35)),
                ("hain", (2.40, 2.75)),
            ],
            "Agent pauses ~1.1s then continues; user channel is silence.",
        )
    )

    # 3) Backchannel-ish: user long turn, short agent filler
    dur = 3.0
    t = np.arange(int(dur * sr), dtype=np.float32) / sr
    user = _tone(300.0, t, ((t >= 0.1) & (t < 2.6)).astype(np.float32))
    agent = _tone(260.0, t, ((t >= 1.2) & (t < 1.55)).astype(np.float32))
    samples.append(
        (
            "03_backchannel",
            np.stack([agent, user], axis=0),
            [("haan", (1.20, 1.50))],
            "User talks continuously; agent short haan (backchannel).",
        )
    )
    return samples


def write_inspect_samples(out_dir: Path, *, text_delay_sec: float = 0.16) -> list[PackedClip]:
    """Write three sample clips + README for local download/inspect."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "train.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()
    synth_provenance = {
        "transcript_backend": "synthetic",
        "transcript_model": "none",
        "timer_backend": "synthetic",
        "timer_model": "none",
        "language": "hi",
        "transcript_text": "",
    }
    clips: list[PackedClip] = []
    lines = [
        "# Moshi Inner Monologue — 3 inspect samples",
        "",
        "Synthetic stereo tones (not real speech) so you can check **layout** without ASR.",
        "",
        "| File | Scenario |",
        "|------|----------|",
    ]
    for stem, stereo, words, desc in make_sample_dialogues():
        clip = save_clip(
            out_dir,
            stem,
            stereo,
            words,
            jsonl_path,
            text_delay_sec=text_delay_sec,
            provenance=synth_provenance,
        )
        clips.append(clip)
        lines.append(f"| `wav/{stem}.wav` + `.json` | {desc} |")
    lines.extend(
        [
            "",
            "## Layout",
            "",
            "- **Left channel** = agent (Moshi)",
            "- **Right channel** = user",
            "- `*.json` = agent word alignments only (`SPEAKER_MAIN`)",
            "- `train.jsonl` = `{\"path\", \"duration\", \"extraction\"}`",
            "",
            "Open the wav in any editor that shows stereo; open the sibling JSON for times.",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {
        "sample_rate": SAMPLE_RATE,
        "mimi_frame_rate_hz": MIMI_FRAME_RATE,
        "text_delay_sec": text_delay_sec,
        "streams": 17,
        "layout": ["W_agent_text", "A_agent_8", "A_user_8"],
        "control_tokens": False,
        "synthetic_tones": True,
        "n_clips": len(clips),
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # Portable paths inside the folder / zip
    rows = []
    for clip in clips:
        rows.append(
            {
                "path": f"wav/{clip.wav_path.name}",
                "duration": clip.duration,
                "extraction": synth_provenance,
            }
        )
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return clips


def maybe_encode_mimi(
    clips: list[PackedClip],
    out_dir: Path,
    text_delay_sec: float,
    device: str,
    hf_repo: str,
) -> None:
    import torch

    from ...architecture import MoshiSystem
    from .inner_monologue import stack_joint_sequence
    from .interleaver import Interleaver

    system = MoshiSystem.from_pretrained(hf_repo=hf_repo, device=device, load_weight=True)
    lm = system.lm
    interleaver = Interleaver(
        system.text_tokenizer,
        system.frame_rate,
        lm.text_padding_token_id,
        lm.end_of_text_padding_id,
        lm.zero_token_id,
        keep_main_only=True,
        audio_delay=text_delay_sec,
        device=device,
    )
    dump_dir = out_dir / "tokens"
    dump_dir.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        wav, sr = read_pcm16_wav(clip.wav_path)
        wav = resample_linear(wav, sr, SAMPLE_RATE)
        tensor = torch.from_numpy(wav).to(device=device, dtype=torch.float32)
        audio_codes = system.encode_stereo(tensor)  # [1, 16, S]
        payload = json.loads(clip.json_path.read_text(encoding="utf-8"))
        alignments = [
            (a[0], (float(a[1][0]), float(a[1][1])), a[2]) for a in payload["alignments"]
        ]
        duration = audio_codes.shape[-1] / system.frame_rate
        text = interleaver.prepare_item(alignments, duration)
        t = audio_codes.shape[-1]
        if text.shape[-1] < t:
            text = torch.nn.functional.pad(text, (0, t - text.shape[-1]), value=lm.zero_token_id)
        else:
            text = text[..., :t]
        joint = stack_joint_sequence(text, audio_codes[:, :8], audio_codes[:, 8:])
        torch.save(
            {
                "codes": joint.cpu(),
                "path": str(clip.wav_path),
                "layout": ["text", "agent_mimi_8", "user_mimi_8"],
            },
            dump_dir / f"{clip.wav_path.stem}.pt",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Moshi Inner Monologue train jsonl (dual-channel + agent ASR)."
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--stereo-dir", type=Path, default=None, help="Folder of wavs (stereo preferred).")
    parser.add_argument("--indicvoices", action="store_true", help="Pull conversational rows from IndicVoices.")
    parser.add_argument(
        "--indicvoices-config",
        default=",".join(INDICVOICES_FOCUS),
        help="Comma-separated HF configs (default: hindi,telugu,kannada,tamil).",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Optional per-language cap for smoke tests. Default: no limit (all conversation rows).",
    )
    parser.add_argument("--dummy", action="store_true", help="Write one synthetic clip (no ASR).")
    parser.add_argument(
        "--samples",
        action="store_true",
        help="Write 3 inspectable synthetic clips (overlap / pause / backchannel).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="ASR language override (e.g. hi). Default: derive from each IndicVoices config.",
    )
    parser.add_argument(
        "--asr-backend",
        default="indic-conformer",
        choices=["indic-conformer", "whisper"],
        help="Agent ASR: IndicConformer+align (default) or Whisper.",
    )
    parser.add_argument(
        "--asr-model",
        default=None,
        help="Whisper size (e.g. large-v3) or HF Conformer repo id. Default depends on backend.",
    )
    parser.add_argument("--alignments-json", type=Path, default=None, help="Reuse word times instead of ASR.")
    parser.add_argument("--agent-channel", type=int, default=0, help="Agent = Moshi = left.")
    parser.add_argument("--user-channel", type=int, default=1)
    parser.add_argument(
        "--allow-silent-user",
        action="store_true",
        help="Mono without diarize → agent + silence (legacy single-stream).",
    )
    parser.add_argument(
        "--diarize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mono → pyannote 2-speaker gate to stereo (default: on). Use --no-diarize to disable.",
    )
    parser.add_argument(
        "--text-delay-frames",
        type=int,
        default=2,
        help="Shift Inner Monologue text this many 80 ms steps ahead of agent audio.",
    )
    parser.add_argument("--encode-mimi", action="store_true", help="Also dump 17-stream .pt (needs GPU + weights).")
    parser.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-bf16", help="Moshi weights for --encode-mimi.")
    parser.add_argument("--device", default="cuda", help="Device for ASR / Mimi encode.")
    parser.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload out/ to a private HF dataset (default: on). Use --no-push-to-hub to skip.",
    )
    parser.add_argument(
        "--hf-dataset",
        default=None,
        help="Dataset repo id or name (default: <you>/moshi-indic-inner-monologue).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    text_delay_sec = args.text_delay_frames / MIMI_FRAME_RATE
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.samples:
        clips = write_inspect_samples(out_dir, text_delay_sec=text_delay_sec)
        logger.info("Wrote %d inspect samples under %s", len(clips), out_dir)
        # Zip for easy download
        import zipfile

        zip_path = out_dir / "moshi_im_samples.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out_dir.rglob("*")):
                if p.is_file() and p.name != zip_path.name:
                    zf.write(p, p.relative_to(out_dir).as_posix())
        logger.info("Zip: %s", zip_path)
        print(zip_path)
        if args.push_to_hub:
            from .push import push_private_dataset

            push_private_dataset(out_dir, hf_dataset=args.hf_dataset, include_tokens=False)
        return 0

    jsonl_path = out_dir / "train.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    meta = {
        "sample_rate": SAMPLE_RATE,
        "mimi_frame_rate_hz": MIMI_FRAME_RATE,
        "text_delay_frames": args.text_delay_frames,
        "text_delay_sec": text_delay_sec,
        "streams": 17,
        "layout": ["W_agent_text", "A_agent_8", "A_user_8"],
        "control_tokens": False,
        "inner_monologue_user_text": False,
        "asr_backend": args.asr_backend,
        "language": args.language,
        "indicvoices_configs": parse_indicvoices_configs(args.indicvoices_config)
        if args.indicvoices
        else [],
        "max_clips_per_lang": args.max_clips,
        "diarize": args.diarize,
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    clips: list[PackedClip] = []
    common = dict(
        language=args.language or "hi",
        asr_backend=args.asr_backend,
        asr_model=args.asr_model,
        asr_device=args.device,
        agent_channel=args.agent_channel,
        user_channel=args.user_channel,
        allow_silent_user=args.allow_silent_user,
        diarize=args.diarize,
        text_delay_sec=text_delay_sec,
        out_dir=out_dir,
        jsonl_path=jsonl_path,
    )

    if args.dummy:
        stereo, words = make_dummy_dialogue()
        clips.append(
            save_clip(
                out_dir,
                "dummy_dialogue",
                stereo,
                words,
                jsonl_path,
                text_delay_sec=text_delay_sec,
                provenance={
                    "transcript_backend": "synthetic",
                    "transcript_model": "none",
                    "timer_backend": "synthetic",
                    "timer_model": "none",
                    "language": "hi",
                    "transcript_text": "",
                },
            )
        )

    extra_align = load_alignments_json(args.alignments_json) if args.alignments_json else None

    if args.stereo_dir:
        from .asr import AlignmentError
        from .diarize import DiarizationSkip

        n = 0
        for stem, wav, sr in iter_stereo_dir(args.stereo_dir):
            try:
                clip = process_stereo_array(
                    stem,
                    wav,
                    sr,
                    alignments=extra_align,
                    **common,
                )
            except (DiarizationSkip, AlignmentError) as err:
                logger.warning("Skip %s: %s", stem, err)
                continue
            clips.append(clip)
            n += 1
            if args.max_clips is not None and n >= args.max_clips:
                break
        logger.info("Packed %d files from %s", n, args.stereo_dir)

    if args.indicvoices:
        configs = parse_indicvoices_configs(args.indicvoices_config)
        iv = process_indicvoices_many(
            language_configs=configs,
            split=args.split,
            max_clips_per_lang=args.max_clips,
            streaming=args.streaming,
            language_override=args.language,
            common=common,
        )
        clips.extend(iv)
        logger.info(
            "Packed %d IndicVoices conversation rows total across %s",
            len(iv),
            ",".join(configs),
        )

    if not clips:
        raise SystemExit("Nothing written. Pass --dummy, --stereo-dir, or --indicvoices.")

    if args.encode_mimi and clips:
        maybe_encode_mimi(clips, out_dir, text_delay_sec, args.device, args.hf_repo)

    n_rows = sum(1 for _ in jsonl_path.open()) if jsonl_path.exists() else 0
    hours = 0.0
    if jsonl_path.exists():
        hours = sum(json.loads(l)["duration"] for l in jsonl_path.open()) / 3600.0
    logger.info("Wrote %s (%d clips, %.3f h)", jsonl_path, n_rows, hours)
    print(jsonl_path)

    if args.push_to_hub:
        from .push import push_private_dataset

        push_private_dataset(
            out_dir,
            hf_dataset=args.hf_dataset,
            include_tokens=bool(args.encode_mimi),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
