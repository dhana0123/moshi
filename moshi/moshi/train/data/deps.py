# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Preflight checks for ``moshi.train.data.prepare`` optional deps."""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger("moshi.prepare.deps")

# (import_name, pip_extra_hint)
_REQUIRED: list[tuple[str, str]] = [
    ("datasets", "datasets>=4.4.0"),
    ("pyarrow", "pyarrow>=21.0.0"),
    ("soundfile", "soundfile"),
    ("huggingface_hub", "huggingface-hub"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("pyannote.audio", "pyannote.audio>=3.1,<4"),
    ("nemo.collections.asr", "nemo_toolkit[asr] or AI4Bharat/NeMo nemo-v2"),
    ("torchaudio", "torchaudio"),
    ("whisperx", "whisperx"),
    ("transformers", "transformers"),
]


def missing_data_packages(*, diarize: bool = True, asr_backend: str = "indic-conformer") -> list[str]:
    missing: list[str] = []
    broken: list[str] = []
    for mod, pip_name in _REQUIRED:
        if mod == "pyannote.audio" and not diarize:
            continue
        if mod.startswith("nemo") and asr_backend != "indic-conformer":
            continue
        if mod == "whisperx" and asr_backend not in {"indic-conformer", "whisper"}:
            # whisperx only required for indic-conformer timing; whisper backend has own times
            if asr_backend != "indic-conformer":
                continue
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pip_name)
        except Exception as exc:
            broken.append(f"{pip_name} ({type(exc).__name__}: {exc})")
    return missing + broken


def require_data_deps(*, diarize: bool = True, asr_backend: str = "indic-conformer") -> None:
    problems = missing_data_packages(diarize=diarize, asr_backend=asr_backend)
    if not problems:
        logger.info("Data deps OK (%d packages checked).", len(_REQUIRED))
        return
    uniq = sorted(set(problems))
    raise ImportError(
        "Missing or broken packer dependencies:\n  - "
        + "\n  - ".join(uniq)
        + "\n\nFix common pyarrow/datasets clash:\n"
        "  pip install -U 'datasets>=4.4.0' 'pyarrow>=21'\n"
        "  # OR keep old datasets: pip install 'pyarrow==20.0.0'\n\n"
        "Or install extras:\n"
        "  cd moshi/moshi && pip install -e \".[data]\" --upgrade-strategy only-if-needed\n"
        "  pip install -r requirements-data.txt\n"
        "then re-run prepare."
    )
