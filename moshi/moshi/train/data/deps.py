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
    problems: list[str] = []
    for mod, pip_name in _REQUIRED:
        if mod == "pyannote.audio" and not diarize:
            continue
        if mod.startswith("nemo") and asr_backend != "indic-conformer":
            continue
        # WhisperX only for IndicConformer word timing
        if mod == "whisperx" and asr_backend != "indic-conformer":
            continue
        try:
            importlib.import_module(mod)
        except Exception as exc:
            problems.append(f"{pip_name}  [{type(exc).__name__}: {exc}]")
    return problems


def require_data_deps(*, diarize: bool = True, asr_backend: str = "indic-conformer") -> None:
    problems = missing_data_packages(diarize=diarize, asr_backend=asr_backend)
    if not problems:
        logger.info("Data deps OK (%d packages checked).", len(_REQUIRED))
        return
    raise ImportError(
        "Missing or broken packer dependencies:\n  - "
        + "\n  - ".join(problems)
        + "\n\nInstall the missing ones, e.g.:\n"
        "  pip install 'pyannote.audio>=3.1,<4' transformers\n"
        "  pip install -U 'datasets>=4.4.0' 'pyarrow>=21'\n"
        "  # if torch got replaced: pip install torch torchaudio "
        "--index-url https://download.pytorch.org/whl/cu128\n\n"
        "Or: pip install -e \".[data]\" --upgrade-strategy only-if-needed\n"
        "then re-run prepare."
    )
