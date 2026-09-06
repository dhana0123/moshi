# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import shutil
from pathlib import Path

import safetensors.torch
import torch

from .distributed import get_rank, get_world_size
from .utils import TrainState

logger = logging.getLogger("moshi.train")


class Checkpointer:
    def __init__(
        self,
        model: torch.nn.Module,
        state: TrainState,
        run_dir: Path | str,
        config: dict,
        optimizer: torch.optim.Optimizer | None = None,
        num_ckpt_keep: int | None = 3,
    ):
        self.model = model
        self.optimizer = optimizer
        self.state = state
        self.run_dir = Path(run_dir)
        self.num_ckpt_keep = num_ckpt_keep
        self.config = config

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def dst_dir(self) -> Path:
        return self.ckpt_dir / f"checkpoint_{self.state.step:06d}"

    def delete_old_ckpts(self) -> None:
        if self.num_ckpt_keep is None or not self.ckpt_dir.exists():
            return
        saved = sorted([d for d in self.ckpt_dir.iterdir() if d.is_dir()], key=lambda p: p.name)
        for old in saved[: max(0, len(saved) - self.num_ckpt_keep)]:
            shutil.rmtree(old, ignore_errors=True)

    @torch.no_grad()
    def save_checkpoint(self, dtype: torch.dtype = torch.bfloat16) -> None:
        if get_rank() != 0:
            return
        self.dst_dir.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().to(dtype=dtype, device="cpu") for k, v in self.model.state_dict().items()}
        safetensors.torch.save_file(state, self.dst_dir / "consolidated.safetensors")
        (self.dst_dir / "config.json").write_text(json.dumps(self.config, indent=2, default=str))
        logger.info("Saved checkpoint to %s", self.dst_dir)
        self.delete_old_ckpts()
        _ = get_world_size  # keep import used
