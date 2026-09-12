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
) -> dict:
    """Sidecar JSON with acoustic word times. Text delay is a train-time shift."""
    return {
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


def write_sidecar(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(jsonl_path: Path, wav_path: Path, duration: float) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"path": str(wav_path.resolve()), "duration": float(duration)}
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def is_conversation_row(row: dict) -> bool:
    blob = f"{row.get('task_name', '')} {row.get('scenario', '')}".lower()
    return any(m in blob for m in CONVERSATION_MARKERS)


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
) -> PackedClip:
    wav_path = out_dir / "wav" / f"{stem}.wav"
    json_path = out_dir / "wav" / f"{stem}.json"
    write_pcm16_wav(wav_path, stereo, sample_rate)
    duration = stereo.shape[-1] / sample_rate
    write_sidecar(json_path, alignments_payload(words, text_delay_sec=text_delay_sec))
    append_jsonl(jsonl_path, wav_path, duration)
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
    text_delay_sec: float,
    out_dir: Path,
    jsonl_path: Path,
) -> PackedClip:
    from .asr import transcribe_agent_words

    x = np.asarray(wav, dtype=np.float32)
    if x.ndim == 2 and x.shape[0] not in (1, 2) and x.shape[1] in (1, 2):
        x = x.T
    if x.ndim == 1:
        x = x[None]
    if x.shape[0] == 1:
        if not allow_silent_user:
            raise ValueError(
                f"{name}: mono file. Pass --allow-silent-user for single-stream "
                "pretrain format, or provide stereo / dual-mono."
            )
        stereo = pack_stereo(x[0], np.zeros_like(x[0]), sr)
    else:
        if agent_channel >= x.shape[0] or user_channel >= x.shape[0]:
            raise ValueError(f"{name}: not enough channels ({x.shape[0]})")
        stereo = pack_stereo(x[agent_channel], x[user_channel], sr)
    agent = stereo[0]
    if alignments is None:
        words = transcribe_agent_words(
            agent,
            SAMPLE_RATE,
            language=language,
            backend=asr_backend,
            asr_model=asr_model,
            device=asr_device,
        )
    else:
        words = alignments
    return save_clip(out_dir, name, stereo, words, jsonl_path, text_delay_sec=text_delay_sec)


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
    if not streaming:
        ds = ds.cast_column("audio_filepath", Audio(sampling_rate=SAMPLE_RATE))
    clips: list[PackedClip] = []
    for i, row in enumerate(ds):
        if not is_conversation_row(row):
            continue
        audio = row.get("audio_filepath") or row.get("audio")
        if audio is None:
            continue
        if isinstance(audio, dict):
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio.get("sampling_rate") or SAMPLE_RATE)
        else:
            continue
        stem = f"{language_config}_{split}_{i:08d}"
        try:
            clips.append(process_stereo_array(stem, arr, sr, alignments=None, **kwargs))
        except Exception:
            logger.exception("Failed IndicVoices row %s", stem)
            continue
        if max_clips is not None and len(clips) >= max_clips:
            break
    return clips


def make_dummy_dialogue(duration_sec: float = 2.56, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, list[tuple[str, tuple[float, float]]]]:
    """Deterministic dual-channel clip for pipeline tests (agent then overlap)."""
    t = np.arange(int(duration_sec * sr), dtype=np.float32) / sr
    agent = 0.08 * np.sin(2 * np.pi * 220 * t)
    user = 0.08 * np.sin(2 * np.pi * 330 * t)
    # Agent talks 0.2–1.2 s; user 0.9–2.0 s (natural overlap, no state token).
    agent_gate = ((t >= 0.2) & (t < 1.2)).astype(np.float32)
    user_gate = ((t >= 0.9) & (t < 2.0)).astype(np.float32)
    stereo = np.stack([agent * agent_gate, user * user_gate], axis=0)
    words = [
        ("hello", (0.20, 0.55)),
        ("there", (0.58, 1.10)),
    ]
    return stereo, words


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
    parser.add_argument("--indicvoices-config", default="hindi", help="HF config name, e.g. hindi.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--dummy", action="store_true", help="Write one synthetic clip (no ASR).")
    parser.add_argument("--language", default="hi", help="ASR language code.")
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
        help="Mono → agent + silence (single-stream pretrain, paper §4.2).",
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
        action="store_true",
        help="Upload out/ to a private HF dataset (HF_TOKEN / huggingface-cli login).",
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
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    clips: list[PackedClip] = []
    common = dict(
        language=args.language,
        asr_backend=args.asr_backend,
        asr_model=args.asr_model,
        asr_device=args.device,
        agent_channel=args.agent_channel,
        user_channel=args.user_channel,
        allow_silent_user=args.allow_silent_user,
        text_delay_sec=text_delay_sec,
        out_dir=out_dir,
        jsonl_path=jsonl_path,
    )

    if args.dummy:
        stereo, words = make_dummy_dialogue()
        clips.append(
            save_clip(out_dir, "dummy_dialogue", stereo, words, jsonl_path, text_delay_sec=text_delay_sec)
        )

    extra_align = load_alignments_json(args.alignments_json) if args.alignments_json else None

    if args.stereo_dir:
        n = 0
        for stem, wav, sr in iter_stereo_dir(args.stereo_dir):
            clip = process_stereo_array(
                stem,
                wav,
                sr,
                alignments=extra_align,
                **common,
            )
            clips.append(clip)
            n += 1
            if args.max_clips is not None and n >= args.max_clips:
                break
        logger.info("Packed %d files from %s", n, args.stereo_dir)

    if args.indicvoices:
        iv = process_indicvoices(
            language_config=args.indicvoices_config,
            split=args.split,
            max_clips=args.max_clips,
            streaming=args.streaming,
            **common,
        )
        clips.extend(iv)
        logger.info("Packed %d IndicVoices conversation rows", len(iv))

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
