# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Word-level ASR timestamps for duplex scoring (Parakeet)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_parakeet_asr_model = None


def get_time_aligned_transcription(audio_path: str) -> List[Dict[str, Any]]:
    """Transcribe audio with nvidia/parakeet-tdt-0.6b-v2 and return word timestamps."""
    global _parakeet_asr_model
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as err:
        raise ImportError(
            "nemo_toolkit[asr] is required for duplex eval ASR. "
            "Install with: pip install -e '.[eval]'"
        ) from err

    if _parakeet_asr_model is None:
        _parakeet_asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2"
        ).cuda()
        _parakeet_asr_model.change_attention_model("rel_pos_local_attn", [128, 128])
        _parakeet_asr_model.change_subsampling_conv_chunking_factor(1)

    try:
        asr_outputs = _parakeet_asr_model.transcribe([audio_path], timestamps=True)
    except Exception as err:
        logger.warning("Failed to transcribe audio %s: %s", audio_path, err)
        return []

    result = asr_outputs[0]
    # NeMo may return Hypothesis or nested list depending on version.
    if isinstance(result, (list, tuple)):
        result = result[0]
    word_timestamps = result.timestep["word"]

    timestamps = []
    for w in word_timestamps:
        timestamps.append(
            {
                "text": w["word"],
                "timestamp": [w["start"], w["end"]],
            }
        )
    return timestamps
