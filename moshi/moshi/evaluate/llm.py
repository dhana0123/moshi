# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal OpenAI-compatible client for User Interruption judging."""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        system_prompt: str,
        *,
        base_url: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        from openai import OpenAI

        self.model_name = model_name if model_name is not None else os.environ.get(
            "LLM_MODEL_NAME", "gpt-4-turbo"
        )
        self.system_prompt = system_prompt
        self.temperature = 1.0

        resolved_base = base_url or os.environ.get("LLM_BASE_URL")
        if not resolved_base:
            raise RuntimeError(
                "LLM_BASE_URL is required for User Interruption judging "
                "(OpenAI-compatible chat completions endpoint)."
            )
        resolved_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "LLM_API_KEY is required for User Interruption judging."
            )
        self.client = OpenAI(base_url=resolved_base, api_key=resolved_key)

    def generate(
        self,
        prompt: str,
        context: str = "",
        max_new_tokens: int = 512,
        stop_token: str | None = None,
        **kwargs,
    ) -> str:
        messages = []
        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt + context}],
            }
        )
        t0 = time.monotonic()
        create_kwargs = dict(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=self.temperature,
            stop=[stop_token] if stop_token is not None else None,
        )
        # Drop unsupported kwargs from callers (e.g. seed).
        create_kwargs.update({k: v for k, v in kwargs.items() if k in ("seed",)})
        try:
            response = self.client.chat.completions.create(**create_kwargs)
        except TypeError:
            # Some backends reject seed / stop=None.
            create_kwargs.pop("seed", None)
            if create_kwargs.get("stop") is None:
                create_kwargs.pop("stop", None)
            response = self.client.chat.completions.create(**create_kwargs)

        text = response.choices[0].message.content
        if text:
            text = text.strip()
        else:
            logger.warning(
                "LLM returned empty content (finish_reason=%s)",
                response.choices[0].finish_reason,
            )
            text = ""
        logger.info("LLM response generation took %.2fs", time.monotonic() - t0)
        return text
