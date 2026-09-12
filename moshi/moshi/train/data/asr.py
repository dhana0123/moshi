# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Agent-channel ASR for Inner Monologue packer.

Strict path: IndicConformer (CTC) for transcript + **WhisperX** forced alignment
for word times. No MMS / even-split fallbacks — missing aligner fails the clip.

Do **not** use NeMo Forced Aligner on IndicConformer (tokenizer mismatch).
"""

from __future__ import annotations

import logging
import re
import tempfile
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger("moshi.prepare.asr")

ASR_SAMPLE_RATE = 16_000
HINDI_CONFORMER = "ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large"
MULTI_CONFORMER = "ai4bharat/indic-conformer-600m-multilingual"

# Explicit WhisperX wav2vec2 align models for our focus languages.
WHISPERX_ALIGN_MODELS: dict[str, str] = {
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "ta": "manandey/wav2vec2-large-xlsr-tamil",
    "kn": "amoghsgopadi/wav2vec2-large-xlsr-kn",
}

WordSpan = tuple[str, tuple[float, float]]

_conformer_cache: dict[str, object] = {}


class AlignmentError(RuntimeError):
    """Strict timing failed (WhisperX missing or align model failed)."""


@dataclass
class AsrProvenance:
    transcript_backend: str
    transcript_model: str
    timer_backend: str
    timer_model: str
    language: str
    transcript_text: str = ""


@dataclass
class AsrResult:
    words: list[WordSpan]
    provenance: AsrProvenance

    def provenance_dict(self) -> dict:
        return asdict(self.provenance)


def resample_mono(x: np.ndarray, src_sr: int, dst_sr: int = ASR_SAMPLE_RATE) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=0) if x.shape[0] < x.shape[-1] else x.mean(axis=-1)
    if src_sr == dst_sr:
        return x
    n_src = x.shape[-1]
    n_dst = int(round(n_src * dst_sr / src_sr))
    if n_dst <= 1:
        return np.zeros(max(n_dst, 0), dtype=np.float32)
    src_t = np.linspace(0.0, 1.0, n_src, endpoint=False)
    dst_t = np.linspace(0.0, 1.0, n_dst, endpoint=False)
    return np.interp(dst_t, src_t, x).astype(np.float32)


def split_words(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", text.strip()) if w]


def conformer_repo_for_language(language: str) -> str:
    lang = (language or "hi").lower().replace("_", "-").split("-")[0]
    if lang in {"hi", "hin", "hindi"}:
        return HINDI_CONFORMER
    return MULTI_CONFORMER


def language_id_for_nemo(language: str) -> str:
    lang = (language or "hi").lower().replace("_", "-").split("-")[0]
    aliases = {"hin": "hi", "hindi": "hi", "tam": "ta", "tel": "te", "ben": "bn", "mar": "mr", "kan": "kn"}
    return aliases.get(lang, lang)


def whisperx_align_model_for_language(language: str) -> str:
    lang = language_id_for_nemo(language)
    if lang not in WHISPERX_ALIGN_MODELS:
        raise AlignmentError(
            f"No WhisperX align model mapped for language={lang!r}. "
            f"Supported: {sorted(WHISPERX_ALIGN_MODELS)}"
        )
    return WHISPERX_ALIGN_MODELS[lang]


def load_indic_conformer(repo: str, device: str = "cuda"):
    try:
        import nemo.collections.asr as nemo_asr
        import torch
    except ImportError as exc:
        raise ImportError(
            "IndicConformer needs nemo_toolkit[asr]. "
            "Install: pip install 'nemo_toolkit[asr]' (or moshi[eval/data])."
        ) from exc

    key = f"{repo}|{device}"
    if key in _conformer_cache:
        return _conformer_cache[key]

    logger.info("Loading IndicConformer %s on %s …", repo, device)
    model = nemo_asr.models.ASRModel.from_pretrained(repo)
    if hasattr(model, "cur_decoder"):
        model.cur_decoder = "ctc"
    model.freeze()
    model.eval()
    if device.startswith("cuda") and torch.cuda.is_available():
        model = model.to(torch.device(device))
    else:
        model = model.cpu()
        device = "cpu"
    _conformer_cache[key] = model
    logger.info("IndicConformer ready: %s", repo)
    return model


def _write_temp_wav_16k(audio_16k: np.ndarray) -> str:
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    pcm = np.clip(audio_16k, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(ASR_SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return path


def transcribe_indic_conformer(
    mono: np.ndarray,
    sample_rate: int,
    language: str = "hi",
    *,
    device: str = "cuda",
    model_name: str | None = None,
) -> tuple[str, str]:
    """Return ``(transcript, model_repo)`` from IndicConformer CTC."""
    repo = model_name or conformer_repo_for_language(language)
    model = load_indic_conformer(repo, device=device)
    audio_16k = resample_mono(mono, sample_rate, ASR_SAMPLE_RATE)
    lang_id = language_id_for_nemo(language)
    path = _write_temp_wav_16k(audio_16k)

    try:
        kwargs: dict = {"batch_size": 1}
        try:
            out = model.transcribe([path], language_id=lang_id, **kwargs)
        except TypeError:
            out = model.transcribe([path], **kwargs)
    finally:
        Path(path).unlink(missing_ok=True)

    return _extract_transcript(out).strip(), repo


def _extract_transcript(out) -> str:
    if out is None:
        return ""
    if isinstance(out, str):
        return out
    if isinstance(out, (list, tuple)) and out:
        first = out[0]
        if isinstance(first, str):
            return first
        if hasattr(first, "text"):
            return str(first.text)
        if isinstance(first, (list, tuple)) and first:
            return str(first[0])
        return str(first)
    if hasattr(out, "text"):
        return str(out.text)
    return str(out)


_align_cache: dict[str, tuple[object, object]] = {}


def load_whisperx_align_model(language: str, device: str = "cuda"):
    """Download/cache WhisperX wav2vec2 aligner for ``language``."""
    try:
        import whisperx
    except ImportError as exc:
        raise AlignmentError(
            "WhisperX is required for word timing. Install: pip install whisperx"
        ) from exc

    lang = language_id_for_nemo(language)
    align_repo = whisperx_align_model_for_language(lang)
    key = f"{lang}|{align_repo}|{device}"
    if key in _align_cache:
        return _align_cache[key][0], _align_cache[key][1], align_repo

    logger.info("Loading WhisperX align model %s (lang=%s) …", align_repo, lang)
    model_a, metadata = whisperx.load_align_model(
        language_code=lang,
        device=device,
        model_name=align_repo,
    )
    _align_cache[key] = (model_a, metadata)
    logger.info("WhisperX align ready: %s", align_repo)
    return model_a, metadata, align_repo


def align_words_whisperx(
    mono: np.ndarray,
    sample_rate: int,
    transcript: str,
    language: str,
    *,
    device: str = "cuda",
) -> tuple[list[WordSpan], str]:
    """Strict WhisperX align. Raises AlignmentError on failure."""
    try:
        import whisperx
    except ImportError as exc:
        raise AlignmentError(
            "WhisperX is required for word timing. Install: pip install whisperx"
        ) from exc

    words = split_words(transcript)
    if not words:
        return [], whisperx_align_model_for_language(language)

    lang = language_id_for_nemo(language)
    model_a, metadata, align_repo = load_whisperx_align_model(lang, device=device)
    audio = resample_mono(mono, sample_rate, ASR_SAMPLE_RATE)
    duration = float(len(audio) / ASR_SAMPLE_RATE)
    segments = [{"start": 0.0, "end": duration, "text": transcript}]

    try:
        aligned = whisperx.align(
            segments,
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
    except Exception as exc:
        raise AlignmentError(
            f"WhisperX align failed for language={lang} model={align_repo}: {exc}"
        ) from exc

    out: list[WordSpan] = []
    for seg in aligned.get("segments", []) or []:
        for w in seg.get("words", []) or []:
            text = str(w.get("word", w.get("text", ""))).strip()
            if not text or "start" not in w or "end" not in w:
                continue
            out.append((text, (float(w["start"]), float(w["end"]))))
    if not out:
        raise AlignmentError(
            f"WhisperX returned no word times (language={lang}, model={align_repo})"
        )
    return out, align_repo


def align_words(
    mono: np.ndarray,
    sample_rate: int,
    transcript: str,
    language: str,
    *,
    device: str = "cuda",
) -> tuple[list[WordSpan], str]:
    """Strict: WhisperX only."""
    return align_words_whisperx(mono, sample_rate, transcript, language, device=device)


def transcribe_words_indic_conformer(
    mono: np.ndarray,
    sample_rate: int,
    language: str = "hi",
    *,
    device: str = "cuda",
    model_name: str | None = None,
) -> AsrResult:
    text, repo = transcribe_indic_conformer(
        mono, sample_rate, language, device=device, model_name=model_name
    )
    if not text.strip():
        raise AlignmentError("IndicConformer returned empty transcript")
    spans, align_repo = align_words(mono, sample_rate, text, language, device=device)
    return AsrResult(
        words=spans,
        provenance=AsrProvenance(
            transcript_backend="indic-conformer",
            transcript_model=repo,
            timer_backend="whisperx",
            timer_model=align_repo,
            language=language_id_for_nemo(language),
            transcript_text=text,
        ),
    )


def transcribe_words_whisper(
    mono: np.ndarray,
    sample_rate: int,
    language: str,
    model_name: str = "large-v3",
) -> AsrResult:
    """Whisper backend with its own word timestamps (timer = same Whisper model)."""
    audio = resample_mono(mono, sample_rate, ASR_SAMPLE_RATE)
    timer_name = model_name
    words: list[WordSpan] = []
    try:
        import whisper_timestamped as whisper_ts

        model = whisper_ts.load_model(model_name)
        out = whisper_ts.transcribe(model, audio, language=language, verbose=False)
        for seg in out.get("segments", []):
            for word in seg.get("words", []):
                text = str(word.get("text", "")).strip()
                if not text:
                    continue
                words.append((text, (float(word["start"]), float(word["end"]))))
        timer_name = f"whisper-timestamped/{model_name}"
    except ImportError:
        try:
            import whisper
        except ImportError as exc:
            raise ImportError(
                "Install whisper-timestamped or openai-whisper for --asr-backend whisper"
            ) from exc
        model = whisper.load_model(model_name)
        result = model.transcribe(audio, language=language, word_timestamps=True, verbose=False)
        for seg in result.get("segments", []):
            for word in seg.get("words", []):
                text = str(word.get("word", word.get("text", ""))).strip()
                if not text:
                    continue
                words.append((text, (float(word["start"]), float(word["end"]))))
        timer_name = f"openai-whisper/{model_name}"
    if not words:
        raise AlignmentError(f"Whisper returned no word times (model={timer_name})")
    return AsrResult(
        words=words,
        provenance=AsrProvenance(
            transcript_backend="whisper",
            transcript_model=timer_name,
            timer_backend="whisper",
            timer_model=timer_name,
            language=language_id_for_nemo(language),
            transcript_text=" ".join(w for w, _ in words),
        ),
    )


def transcribe_agent_words(
    mono: np.ndarray,
    sample_rate: int,
    *,
    language: str = "hi",
    backend: str = "indic-conformer",
    asr_model: str | None = None,
    device: str = "cuda",
) -> AsrResult:
    """Dispatch ASR backend. ``backend``: indic-conformer | whisper."""
    backend = (backend or "indic-conformer").lower().replace("_", "-")
    if backend in {"indic-conformer", "indicconformer", "conformer"}:
        return transcribe_words_indic_conformer(
            mono,
            sample_rate,
            language,
            device=device,
            model_name=asr_model if asr_model and "/" in asr_model else None,
        )
    if backend == "whisper":
        return transcribe_words_whisper(
            mono,
            sample_rate,
            language,
            model_name=asr_model or "large-v3",
        )
    raise ValueError(f"Unknown --asr-backend {backend!r}; use indic-conformer or whisper")
