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
    ("datasets", "datasets"),
    ("soundfile", "soundfile"),
    ("huggingface_hub", "huggingface-hub"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("pyannote.audio", "pyannote.audio>=3.1"),
    ("nemo.collections.asr", "nemo_toolkit[asr]"),
    ("torchaudio", "torchaudio"),
    ("whisperx", "whisperx"),
    ("transformers", "transformers"),
]


def missing_data_packages(*, diarize: bool = True, asr_backend: str = "indic-conformer") -> list[str]:
    missing: list[str] = []
    for mod, pip_name in _REQUIRED:
        if mod == "pyannote.audio" and not diarize:
            continue
        if mod.startswith("nemo") and asr_backend != "indic-conformer":
            continue
        if mod == "whisperx" and asr_backend != "indic-conformer":
            continue
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pip_name)
    return missing


def require_data_deps(*, diarize: bool = True, asr_backend: str = "indic-conformer") -> None:
    missing = missing_data_packages(diarize=diarize, asr_backend=asr_backend)
    if not missing:
        logger.info("Data deps OK (%d packages checked).", len(_REQUIRED))
        return
    uniq = sorted(set(missing))
    raise ImportError(
        "Missing packer dependencies: "
        + ", ".join(uniq)
        + "\n\nInstall everything with:\n"
        "  cd moshi/moshi && pip install -e \".[data]\"\n"
        "or:\n"
        "  pip install -r requirements-data.txt\n"
        "then re-run prepare."
    )
