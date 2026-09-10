# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal


PromptStyle = Literal["original", "simplified"]


@dataclass(frozen=True)
class RetrievalProfile:
    id: str
    base_url: str
    model: str
    api_key: str | None
    is_default: bool
    prompt_style: PromptStyle


@dataclass(frozen=True)
class RetrievalEnvConfig:
    """Parsed ``MOSHI_RETRIEVAL_LLMS_JSON`` (profiles only)."""

    profiles: list[RetrievalProfile]


def _parse_profile(obj: Any, profile_default_prompt_style: PromptStyle) -> RetrievalProfile:
    if not isinstance(obj, dict):
        raise ValueError("profile entry must be an object")
    pid = str(obj["id"]).strip()
    base_url = str(obj["base_url"]).strip()
    model = str(obj["model"]).strip()
    if not pid or not base_url or not model:
        raise ValueError("id, base_url, and model are required and must be non-empty")
    api_key = obj.get("api_key")
    if api_key is not None:
        api_key = str(api_key).strip() or None
    is_default = _parse_default_flag(obj.get("default", False))
    if "prompt_style" in obj:
        prompt_style = _parse_prompt_style(obj["prompt_style"])
    elif "reference_prompt" in obj:
        prompt_style = _parse_prompt_style(obj["reference_prompt"])
    else:
        prompt_style = profile_default_prompt_style
    return RetrievalProfile(
        id=pid,
        base_url=base_url,
        model=model,
        api_key=api_key,
        is_default=is_default,
        prompt_style=prompt_style,
    )


def _parse_default_flag(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s == "true":
            return True
        if s == "false":
            return False
    raise ValueError('default must be a boolean or "true"/"false" string')


def _parse_prompt_style(raw: Any) -> PromptStyle:
    if raw is None:
        return "original"
    if not isinstance(raw, str):
        raise ValueError("prompt_style must be a string")
    s = raw.strip().lower()
    if s == "original":
        return "original"
    if s == "simplified":
        return "simplified"
    raise ValueError('prompt_style must be "original" or "simplified"')


def load_retrieval_env() -> RetrievalEnvConfig:
    raw = os.environ.get("MOSHI_RETRIEVAL_LLMS_JSON", "").strip()
    if not raw:
        return RetrievalEnvConfig([])
    data = json.loads(raw)
    profile_default_prompt_style: PromptStyle = "original"
    if isinstance(data, dict):
        if "prompt_style" in data:
            profile_default_prompt_style = _parse_prompt_style(data["prompt_style"])
        if "profiles" not in data:
            raise ValueError(
                'MOSHI_RETRIEVAL_LLMS_JSON object form requires a "profiles" array '
                "(legacy form is a bare JSON array of profile objects)"
            )
        allowed_top = {"profiles", "prompt_style", "reference_prompt"}
        unknown = [k for k in data if k not in allowed_top]
        if unknown:
            raise ValueError(f"MOSHI_RETRIEVAL_LLMS_JSON unknown top-level keys: {', '.join(sorted(unknown))}")
        data = data["profiles"]
    if not isinstance(data, list):
        raise ValueError("MOSHI_RETRIEVAL_LLMS_JSON must be a JSON array or an object with profiles")
    profiles = [_parse_profile(x, profile_default_prompt_style) for x in data]
    ids = [p.id for p in profiles]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate profile id in MOSHI_RETRIEVAL_LLMS_JSON")
    _validate_default_flags(profiles)
    return RetrievalEnvConfig(profiles)


def load_retrieval_profiles_from_env() -> list[RetrievalProfile]:
    return load_retrieval_env().profiles


def _validate_default_flags(profiles: list[RetrievalProfile]) -> None:
    n = len(profiles)
    if n == 0:
        return
    k = sum(1 for p in profiles if p.is_default)
    if n == 1:
        if k > 1:
            raise ValueError('MOSHI_RETRIEVAL_LLMS_JSON: at most one profile may have "default": true')
        return
    if k != 1:
        raise ValueError(
            f'MOSHI_RETRIEVAL_LLMS_JSON: exactly one profile must have "default": true when using '
            f"multiple profiles (found {k})"
        )


def default_profile_id(profiles: list[RetrievalProfile]) -> str:
    """Id of the fallback profile (always invoked alongside the active profile when they differ)."""
    if not profiles:
        raise ValueError("no profiles")
    if len(profiles) == 1:
        return profiles[0].id
    for p in profiles:
        if p.is_default:
            return p.id
    raise ValueError('MOSHI_RETRIEVAL_LLMS_JSON: exactly one profile must have "default": true')
