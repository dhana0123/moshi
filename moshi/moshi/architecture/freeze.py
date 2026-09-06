# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import typing as tp

import torch
from torch import nn

from .components import COMPONENT_ATTRS, canonical_name

if tp.TYPE_CHECKING:
    from .system import MoshiSystem


def get_by_path(root: nn.Module, path: str) -> nn.Module:
    obj: nn.Module = root
    for part in path.split("."):
        obj = getattr(obj, part)
        if obj is None:
            raise AttributeError(f"Component path {path!r} resolved to None at {part!r}")
    return obj


def get_component(system: "MoshiSystem", name: str) -> nn.Module:
    name = canonical_name(name)
    return get_by_path(system, COMPONENT_ATTRS[name])


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad
    if requires_grad:
        module.train()
    else:
        module.eval()


def apply_freeze(system: "MoshiSystem", freeze_map: dict[str, bool] | None) -> None:
    """Apply a freeze map. Shorter keys run first so more specific keys override.

    Example::

        {"mimi": True, "lm.temporal": False}
    """
    if not freeze_map:
        return
    items = sorted(freeze_map.items(), key=lambda kv: canonical_name(kv[0]).count("."))
    for name, frozen in items:
        set_requires_grad(get_component(system, name), requires_grad=not frozen)


def swap_module(system: "MoshiSystem", name: str, module: nn.Module) -> nn.Module:
    """Replace a named component. Returns the previous module."""
    name = canonical_name(name)
    path = COMPONENT_ATTRS[name]
    parent_path, _, attr = path.rpartition(".")
    parent = system if not parent_path else get_by_path(system, parent_path)
    old = getattr(parent, attr)
    setattr(parent, attr, module)
    return old


def trainable_named_parameters(system: "MoshiSystem") -> list[tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in system.named_parameters() if p.requires_grad]
