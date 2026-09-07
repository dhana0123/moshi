# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Short timed train run to estimate sec/step and project ETA for larger step counts.

Uses the real train path (same YAML / batch / duration as a full run) with a small
``max_steps`` so you can decide hours vs days before committing GPU time.

Example::

    python -m moshi.train.bench_eta --config path/to.yaml --warmup 5 --steps 50
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import statistics
from pathlib import Path

from .args import TrainArgs
from .loop import packaged_config, train_from_args
from .utils import format_eta

logger = logging.getLogger("moshi.train.bench_eta")


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def report_eta(
    step_times: list[float],
    *,
    warmup: int,
    measure_steps: int,
    tokens_seen: int,
    project_steps: list[int],
    config_max_steps: int,
) -> None:
    measured = step_times[warmup:] if warmup < len(step_times) else step_times[:]
    if not measured:
        raise SystemExit("No measured steps after warmup; increase --steps or lower --warmup.")

    mean_s = statistics.fmean(measured)
    sorted_m = sorted(measured)
    median_s = statistics.median(sorted_m)
    p90_s = _percentile(sorted_m, 90)

    measure_tokens = 0
    # Approximate tokens/sec from total tokens over measured wall time only when we
    # can attribute tokens to measured steps (uniform per step).
    if len(step_times) > 0 and tokens_seen > 0:
        tokens_per_step = tokens_seen / len(step_times)
        measure_tokens = tokens_per_step * len(measured)
    tokens_per_sec = measure_tokens / sum(measured) if measured and measure_tokens > 0 else 0.0

    print()
    print("=== Moshi train ETA bench ===")
    print(f"measured_steps={len(measured)} warmup={warmup} (total_ran={len(step_times)})")
    print(
        f"ms_per_step mean={mean_s * 1000:.1f} "
        f"median={median_s * 1000:.1f} "
        f"p90={p90_s * 1000:.1f}"
    )
    print(f"tokens_per_sec={tokens_per_sec:.1f}")
    print()
    targets = list(project_steps)
    if config_max_steps not in targets:
        targets.append(config_max_steps)
    for n in targets:
        label = f"config max_steps ({n})" if n == config_max_steps else str(n)
        print(f"ETA @ {label} steps: {format_eta(n * mean_s)}  ({n * mean_s / 3600:.2f} h)")
    print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=str,
        default=str(packaged_config()),
        help="Train YAML (same as moshi-train). Must set train_data.",
    )
    p.add_argument("--warmup", type=int, default=5, help="Steps to skip before measuring (default: 5).")
    p.add_argument("--steps", type=int, default=50, help="Steps to measure after warmup (default: 50).")
    p.add_argument(
        "--project-steps",
        type=int,
        nargs="+",
        default=[1000, 2000, 10000],
        help="Step counts to project ETA for (default: 1000 2000 10000).",
    )
    p.add_argument(
        "--run-dir",
        type=str,
        default="runs/eta_bench",
        help="Scratch run dir (overwritten). Default: runs/eta_bench",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ns = parse_args(argv)
    if ns.warmup < 0 or ns.steps < 1:
        raise SystemExit("--warmup must be >= 0 and --steps must be >= 1")

    args = TrainArgs.load(ns.config)
    if not args.train_data:
        raise SystemExit(
            "Set train_data in the YAML to a jsonl (or directory of jsonl files) before running the ETA bench."
        )

    config_max_steps = args.max_steps
    total = ns.warmup + ns.steps
    args = dataclasses.replace(
        args,
        max_steps=total,
        do_ckpt=False,
        do_eval=False,
        overwrite_run_dir=True,
        run_dir=ns.run_dir,
        log_freq=max(1, min(args.log_freq, 10)),
    )

    logger.info(
        "ETA bench: warmup=%s measure=%s total_steps=%s batch_size=%s duration_sec=%s "
        "project=%s config_max_steps=%s run_dir=%s",
        ns.warmup,
        ns.steps,
        total,
        args.batch_size,
        args.duration_sec,
        ns.project_steps,
        config_max_steps,
        args.run_dir,
    )

    state = train_from_args(args)
    report_eta(
        state.step_times,
        warmup=ns.warmup,
        measure_steps=ns.steps,
        tokens_seen=state.n_seen_tokens,
        project_steps=ns.project_steps,
        config_max_steps=config_max_steps,
    )


if __name__ == "__main__":
    main()
