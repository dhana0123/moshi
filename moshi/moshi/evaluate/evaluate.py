# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Score Moshi FullDuplexBench outputs (duplex categories only).

Expects results produced by ``python -m moshi.evaluate.run_bench``:
one folder per dataset under ``--results-root``, each containing ``*.json``
samples with a co-located ``.wav``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm

from .judge import (
    BackChannelJudge,
    PauseHandlingJudge,
    TurnTakingJudge,
    UserInterruptionJudge,
)
from .score import ScoreResult, score_file, summarize_folder

logger = logging.getLogger(__name__)

DATASET_DUPLEX_JUDGE_MAP = {
    "candor_pause_handling": PauseHandlingJudge,
    "synthetic_pause_handling": PauseHandlingJudge,
    "icc_backchannel": BackChannelJudge,
    "candor_turn_taking": TurnTakingJudge,
    "synthetic_user_interruption": UserInterruptionJudge,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/fullduplex"),
        help="Root directory with one folder per dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only load existing *.score.json files and write folder summaries.",
    )
    parser.add_argument(
        "--force-reeval",
        action="store_true",
        help="Re-run scoring and overwrite existing score files.",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        required=False,
        help="Dataset folder names to process (default: all under results-root).",
    )
    return parser.parse_args()


def _init_duplex_judge(dataset_name: str):
    judge_cls = DATASET_DUPLEX_JUDGE_MAP.get(dataset_name)
    if judge_cls is None:
        return None
    if judge_cls is UserInterruptionJudge:
        if not os.environ.get("LLM_API_KEY") or not os.environ.get("LLM_BASE_URL"):
            logger.error(
                "Dataset %s requires LLM_API_KEY and LLM_BASE_URL; skipping.",
                dataset_name,
            )
            return None
        return judge_cls()
    return judge_cls()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()

    if not results_root.exists():
        logger.error("Results root %s does not exist.", results_root)
        sys.exit(1)

    folders = [p for p in sorted(results_root.glob("*")) if p.is_dir()]
    for folder in folders:
        if args.dataset_names and folder.name not in args.dataset_names:
            continue
        if folder.name not in DATASET_DUPLEX_JUDGE_MAP:
            logger.info("Skipping unknown dataset folder %s", folder.name)
            continue

        duplex_judge = None
        if not args.summary_only:
            duplex_judge = _init_duplex_judge(folder.name)
            if duplex_judge is None and folder.name == "synthetic_user_interruption":
                continue

        logger.info("Processing %s", folder)
        records: List[ScoreResult] = []
        files = sorted(
            [
                path
                for path in folder.rglob("*.json")
                if "score" not in path.name
            ]
        )
        for path in tqdm(files, desc=folder.name):
            output_path = path.with_name(f"{path.stem}.score.json")
            if args.summary_only:
                if not output_path.exists():
                    continue
                try:
                    import json

                    data = json.loads(output_path.read_text(encoding="utf-8"))
                    rec = ScoreResult.from_entry(data.get("raw_data", {}), path)
                    rec.duplex_scores = data.get("scores", {}).get("duplex_scores", {})
                    rec.timestamps = data.get("processed_data", {}).get("timestamps", [])
                    records.append(rec)
                except Exception as err:
                    logger.warning("Failed to load %s: %s", output_path, err)
                continue

            if output_path.exists() and not args.force_reeval:
                try:
                    import json

                    data = json.loads(output_path.read_text(encoding="utf-8"))
                    rec = ScoreResult.from_entry(data.get("raw_data", {}), path)
                    rec.duplex_scores = data.get("scores", {}).get("duplex_scores", {})
                    rec.timestamps = data.get("processed_data", {}).get("timestamps", [])
                    records.append(rec)
                except Exception as err:
                    logger.warning("Failed to load existing score %s: %s", output_path, err)
                continue

            score = score_file(path, output_path, duplex_judge)
            if score is not None:
                records.append(score)

        summary = summarize_folder(folder, records)
        logger.info("Summary for %s: %s", folder.name, summary.get("duplex_scores"))


if __name__ == "__main__":
    main()
