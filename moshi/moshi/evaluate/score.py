# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Duplex-only scoring for FullDuplexBench Moshi outputs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .asr import get_time_aligned_transcription
from .judge import Judge

logger = logging.getLogger(__name__)


def average(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


@dataclass
class ScoreResult:
    input_path: Path | str = ""
    path: Path = Path("")
    model_text: Optional[List[str] | str] = None
    user_text: Optional[List[str] | str] = None

    timestamps: List[Dict[str, Any]] = field(default_factory=list)
    duplex_scores: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_entry(cls, entry: Dict[str, Any], path: Path) -> "ScoreResult":
        input_path = entry.get("input_path", "")
        return cls(
            input_path=Path(input_path) if input_path else Path(""),
            path=path,
            model_text=entry.get("model_text"),
            user_text=entry.get("user_text"),
        )

    def get_timestamps(self) -> None:
        wav = self.path.with_suffix(".wav")
        if not wav.exists():
            logger.warning("Missing wav for %s", self.path)
            self.timestamps = []
            return
        self.timestamps = get_time_aligned_transcription(str(wav))

    def compute_scores(self, duplex_judge: Judge | None) -> None:
        if duplex_judge is not None:
            self.duplex_scores = duplex_judge(
                Path(self.input_path), self.path, self.timestamps
            )
        else:
            self.duplex_scores = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scores": {"duplex_scores": self.duplex_scores},
            "raw_data": {
                "path": str(self.path),
                "input_path": str(self.input_path),
                "model_text": self.model_text,
                "user_text": self.user_text,
            },
            "processed_data": {
                "timestamps": self.timestamps,
            },
        }


def score_file(
    path: Path,
    output_path: Path,
    duplex_judge: Judge | None,
) -> Optional[ScoreResult]:
    try:
        with path.open(encoding="utf-8") as f:
            entry = json.load(f)
    except Exception as err:
        logger.warning("Failed to parse %s: %s", path, err)
        return None

    score_result = ScoreResult.from_entry(entry, path)
    score_result.get_timestamps()
    score_result.compute_scores(duplex_judge)
    output_path.write_text(json.dumps(score_result.to_dict(), indent=2), encoding="utf-8")
    return score_result


def summarize_folder(folder: Path, records: List[ScoreResult]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "folder": str(folder),
        "num_examples": len(records),
        "duplex_scores": {},
    }

    totals: Dict[str, List[float]] = {}
    for rec in records:
        for key, value in rec.duplex_scores.items():
            if value is None:
                continue
            # Booleans and ints/floats that are valid (>= 0 for latency/score metrics;
            # TO is bool so always include).
            if isinstance(value, bool):
                totals.setdefault(key, []).append(float(value))
            elif isinstance(value, (int, float)) and value >= 0:
                totals.setdefault(key, []).append(float(value))

    summary["duplex_scores"] = {
        key: {"avg": average(vals), "num_examples": len(vals)}
        for key, vals in totals.items()
    }

    output_path = folder / "score_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
