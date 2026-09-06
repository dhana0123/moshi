# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable

import torch


def prepare_mixed_precision(params: Iterable[torch.nn.Parameter], param_dtype: torch.dtype, optim_dtype: torch.dtype):
    with torch.no_grad():
        for p in params:
            if p.requires_grad:
                p._mp_param = torch.empty_like(p, dtype=optim_dtype)  # type: ignore
                p._mp_param.copy_(p.to(optim_dtype))  # type: ignore
            p.data = p.data.to(param_dtype)


def upcast_mixed_precision(params: Iterable[torch.nn.Parameter], optim_dtype: torch.dtype):
    with torch.no_grad():
        for p in params:
            if p.requires_grad and p.grad is not None:
                p._temp = p.data  # type: ignore
                p.data = p._mp_param  # type: ignore
                p.grad = p.grad.to(optim_dtype)


def downcast_mixed_precision(params: Iterable[torch.nn.Parameter], param_dtype: torch.dtype):
    with torch.no_grad():
        for p in params:
            if p.requires_grad and p.grad is not None:
                p._temp.copy_(p.data)  # type: ignore
                p.data = p._temp  # type: ignore
                p.grad = p.grad.to(param_dtype)
