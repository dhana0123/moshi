# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import lru_cache

import torch
import torch.distributed as dist

BACKEND = "nccl"


def dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


@lru_cache()
def get_rank() -> int:
    if dist_ready():
        return dist.get_rank()
    return 0


@lru_cache()
def get_world_size() -> int:
    if dist_ready():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def set_device() -> None:
    if not torch.cuda.is_available():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)


def avg_aggregate(metric: float) -> float:
    if not dist_ready():
        return metric
    buffer = torch.tensor([metric], dtype=torch.float32, device="cuda")
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    return buffer[0].item() / get_world_size()


def is_torchrun() -> bool:
    return "TORCHELASTIC_RESTART_COUNT" in os.environ or "LOCAL_RANK" in os.environ
