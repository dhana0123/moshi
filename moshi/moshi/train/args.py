# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path

from ..models.loaders import DEFAULT_REPO

logger = logging.getLogger("moshi.train")


@dataclass
class OptimArgs:
    temporal_lr: float = 2e-6
    depformer_lr: float = 2e-6
    mimi_lr: float = 8e-4
    weight_decay: float = 0.1
    pct_start: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)


@dataclass
class TrainArgs:
    run_dir: str = "runs/moshi"
    train_data: str = ""
    eval_data: str = ""
    shuffle: bool = True
    hf_repo: str = DEFAULT_REPO
    moshi_path: str | None = None
    mimi_path: str | None = None
    tokenizer_path: str | None = None
    config_path: str | None = None
    init: str = "pretrained"  # pretrained | random
    freeze: dict[str, bool] = field(default_factory=lambda: {"mimi": True})
    first_codebook_weight_multiplier: float = 100.0
    text_padding_weight: float = 0.5
    duration_sec: float = 100.0
    batch_size: int = 1
    num_microbatches: int = 1
    max_steps: int = 100
    max_norm: float = 1.0
    log_freq: int = 1
    ckpt_freq: int = 100
    do_ckpt: bool = True
    num_ckpt_keep: int | None = 3
    eval_freq: int = 0
    do_eval: bool = False
    gradient_checkpointing: bool = True
    param_dtype: str = "bfloat16"
    seed: int = 0
    overwrite_run_dir: bool = False
    audio_delay: float = 0.0
    keep_main_only: bool = True
    text_mask_proba: float = 1.0
    optim: OptimArgs = field(default_factory=OptimArgs)

    @classmethod
    def load(cls, path: str | Path) -> "TrainArgs":
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Install pyyaml (`pip install -e '.[train]'`) to load YAML configs.") from exc
        raw = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in fields(cls)}
        extra = set(raw) - known
        if extra:
            logger.warning("Ignoring unknown config keys: %s", extra)
        kwargs = {k: v for k, v in raw.items() if k in known}
        if "optim" in kwargs and isinstance(kwargs["optim"], dict):
            kwargs["optim"] = OptimArgs(**{
                k: v for k, v in kwargs["optim"].items()
                if k in {f.name for f in fields(OptimArgs)}
            })
        return cls(**kwargs)


def default_freeze_map() -> dict[str, bool]:
    return {"mimi": True}
