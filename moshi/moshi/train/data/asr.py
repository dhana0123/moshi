# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Agent-channel ASR for Inner Monologue packer.

Default path: IndicConformer (CTC) for transcript + forced alignment for word
times. Whisper remains a fallback. Do **not** use NeMo Forced Aligner on
IndicConformer checkpoints (tokenizer/vocab mismatch → bad timings).
"""

from __future__ import annotations

import logging
import re
import tempfile
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger("moshi.prepare.asr")

ASR_SAMPLE_RATE = 16_000
HINDI_CONFORMER = "ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large"
MULTI_CONFORMER = "ai4bharat/indic-conformer-600m-multilingual"

WordSpan = tuple[str, tuple[float, float]]

_conformer_cache: dict[str, object] = {}


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


def _even_word_spans(words: list[str], duration_sec: float) -> list[WordSpan]:
    """Last-resort uniform split when forced alignment is unavailable."""
    if not words:
        return []
    if duration_sec <= 0:
        return [(w, (0.0, 0.0)) for w in words]
    step = duration_sec / len(words)
    return [(w, (i * step, (i + 1) * step)) for i, w in enumerate(words)]


def conformer_repo_for_language(language: str) -> str:
    lang = (language or "hi").lower().replace("_", "-").split("-")[0]
    if lang in {"hi", "hin", "hindi"}:
        return HINDI_CONFORMER
    return MULTI_CONFORMER


def language_id_for_nemo(language: str) -> str:
    lang = (language or "hi").lower().replace("_", "-").split("-")[0]
    # NeMo / AI4Bharat short codes
    aliases = {"hin": "hi", "hindi": "hi", "tam": "ta", "tel": "te", "ben": "bn", "mar": "mr"}
    return aliases.get(lang, lang)


def load_indic_conformer(repo: str, device: str = "cuda"):
    try:
        import nemo.collections.asr as nemo_asr
        import torch
    except ImportError as exc:
        raise ImportError(
            "IndicConformer needs nemo_toolkit[asr]. "
            "Install: pip install 'nemo_toolkit[asr]' (or moshi[eval])."
        ) from exc

    key = f"{repo}|{device}"
    if key in _conformer_cache:
        return _conformer_cache[key]

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
) -> str:
    """Return plain transcript from IndicConformer CTC (agent channel)."""
    repo = model_name or conformer_repo_for_language(language)
    model = load_indic_conformer(repo, device=device)
    audio_16k = resample_mono(mono, sample_rate, ASR_SAMPLE_RATE)
    lang_id = language_id_for_nemo(language)
    path = _write_temp_wav_16k(audio_16k)

    try:
        kwargs: dict = {"batch_size": 1}
        # Multilingual checkpoint wants language_id; Hindi large often accepts it too.
        try:
            out = model.transcribe([path], language_id=lang_id, **kwargs)
        except TypeError:
            out = model.transcribe([path], **kwargs)
    finally:
        Path(path).unlink(missing_ok=True)

    text = _extract_transcript(out)
    return text.strip()


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


def align_words_whisperx(
    mono: np.ndarray,
    sample_rate: int,
    transcript: str,
    language: str,
    *,
    device: str = "cuda",
) -> list[WordSpan] | None:
    try:
        import whisperx
    except ImportError:
        return None

    words = split_words(transcript)
    if not words:
        return []
    audio = resample_mono(mono, sample_rate, ASR_SAMPLE_RATE)
    duration = float(len(audio) / ASR_SAMPLE_RATE)
    segments = [{"start": 0.0, "end": duration, "text": transcript}]
    try:
        model_a, metadata = whisperx.load_align_model(language_code=language_id_for_nemo(language), device=device)
        aligned = whisperx.align(
            segments,
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
    except Exception:
        logger.exception("WhisperX align failed; trying next aligner")
        return None

    out: list[WordSpan] = []
    for seg in aligned.get("segments", []) or []:
        for w in seg.get("words", []) or []:
            text = str(w.get("word", w.get("text", ""))).strip()
            if not text or "start" not in w or "end" not in w:
                continue
            out.append((text, (float(w["start"]), float(w["end"]))))
    return out if out else None


def align_words_mms(
    mono: np.ndarray,
    sample_rate: int,
    transcript: str,
    language: str,
) -> list[WordSpan] | None:
    """Torchaudio MMS forced aligner (supports Hindi among others)."""
    try:
        import torch
        import torchaudio
        from torchaudio.pipelines import MMS_FA as bundle
    except ImportError:
        return None

    words = split_words(transcript)
    if not words:
        return []

    try:
        device = torch.device("cpu")
        model = bundle.get_model()
        model.to(device)
        tokenizer = bundle.get_tokenizer()
        aligner = bundle.get_aligner()
        waveform = torch.from_numpy(resample_mono(mono, sample_rate, bundle.sample_rate)).unsqueeze(0)
        with torch.inference_mode():
            emission, _ = model(waveform.to(device))
            token_spans = aligner(emission[0], tokenizer(words))
    except Exception:
        logger.exception("MMS forced align failed")
        return None

    num_frames = emission.size(1)
    ratio = waveform.size(1) / num_frames
    out: list[WordSpan] = []
    for word, spans in zip(words, token_spans):
        if not spans:
            continue
        start = float(spans[0].start * ratio / bundle.sample_rate)
        end = float(spans[-1].end * ratio / bundle.sample_rate)
        if end <= start:
            end = start + 0.02
        out.append((word, (start, end)))
    return out if out else None


def align_words(
    mono: np.ndarray,
    sample_rate: int,
    transcript: str,
    language: str,
    *,
    device: str = "cuda",
) -> list[WordSpan]:
    words = split_words(transcript)
    if not words:
        return []
    duration = float(len(resample_mono(mono, sample_rate)) / ASR_SAMPLE_RATE)
    for fn in (
        lambda: align_words_whisperx(mono, sample_rate, transcript, language, device=device),
        lambda: align_words_mms(mono, sample_rate, transcript, language),
    ):
        spans = fn()
        if spans:
            return spans
    logger.warning("No forced aligner available; using even word spans (install whisperx or torchaudio)")
    return _even_word_spans(words, duration)


def transcribe_words_indic_conformer(
    mono: np.ndarray,
    sample_rate: int,
    language: str = "hi",
    *,
    device: str = "cuda",
    model_name: str | None = None,
) -> list[WordSpan]:
    text = transcribe_indic_conformer(
        mono, sample_rate, language, device=device, model_name=model_name
    )
    if not text.strip():
        return []
    return align_words(mono, sample_rate, text, language, device=device)


def transcribe_words_whisper(
    mono: np.ndarray,
    sample_rate: int,
    language: str,
    model_name: str = "large-v3",
) -> list[WordSpan]:
    """Word times via whisper-timestamped or openai-whisper."""
    audio = resample_mono(mono, sample_rate, ASR_SAMPLE_RATE)
    try:
        import whisper_timestamped as whisper_ts

        model = whisper_ts.load_model(model_name)
        out = whisper_ts.transcribe(model, audio, language=language, verbose=False)
        words: list[WordSpan] = []
        for seg in out.get("segments", []):
            for word in seg.get("words", []):
                text = str(word.get("text", "")).strip()
                if not text:
                    continue
                words.append((text, (float(word["start"]), float(word["end"]))))
        return words
    except ImportError:
        pass
    try:
        import whisper
    except ImportError as exc:
        raise ImportError(
            "Install whisper-timestamped or openai-whisper for --asr-backend whisper, "
            "or use --asr-backend indic-conformer / --alignments-json / --dummy."
        ) from exc
    model = whisper.load_model(model_name)
    result = model.transcribe(audio, language=language, word_timestamps=True, verbose=False)
    words = []
    for seg in result.get("segments", []):
        for word in seg.get("words", []):
            text = str(word.get("word", word.get("text", ""))).strip()
            if not text:
                continue
            words.append((text, (float(word["start"]), float(word["end"]))))
    return words


def transcribe_agent_words(
    mono: np.ndarray,
    sample_rate: int,
    *,
    language: str = "hi",
    backend: str = "indic-conformer",
    asr_model: str | None = None,
    device: str = "cuda",
) -> list[WordSpan]:
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
