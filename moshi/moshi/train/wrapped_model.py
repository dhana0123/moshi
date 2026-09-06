# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Callable

import torch
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
import torch.distributed.fsdp.wrap as torch_wrap

from ..models.lm import LMModel
from ..modules.transformer import StreamingTransformerLayer
from .distributed import get_world_size


def get_fsdp_policy() -> Callable[[torch.nn.Module], bool]:
    return functools.partial(
        torch_wrap.transformer_auto_wrap_policy,
        transformer_layer_cls=(StreamingTransformerLayer,),
    )


def maybe_fsdp(lm: LMModel, param_dtype: torch.dtype) -> FullyShardedDataParallel | LMModel:
    if get_world_size() <= 1:
        return lm
    return FullyShardedDataParallel(
        lm,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        auto_wrap_policy=get_fsdp_policy(),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        use_orig_params=True,
        mixed_precision=None,
    )


def build_param_groups(lm: torch.nn.Module, mimi: torch.nn.Module, optim):
    dep_prefixes = (
        "depformer",
        "linears",
    )
    temporal, depth, mimi_params, other = [], [], [], []
    for name, p in lm.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(dep_prefixes) or ".depformer" in name or name.startswith("linears"):
            depth.append(p)
        elif name.startswith("transformer") or name.startswith("emb") or name.startswith("text_"):
            temporal.append(p)
        else:
            other.append(p)
    for p in mimi.parameters():
        if p.requires_grad:
            mimi_params.append(p)
    groups = []
    if temporal or other:
        groups.append({"params": temporal + other, "lr": optim.temporal_lr})
    if depth:
        groups.append({"params": depth, "lr": optim.depformer_lr})
    if mimi_params:
        groups.append({"params": mimi_params, "lr": optim.mimi_lr})
    if not groups:
        raise RuntimeError("No trainable parameters. Check the freeze map.")
    return groups
