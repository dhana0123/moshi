# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from openai import OpenAI
import time
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        system_prompt: str,
        prompt: str,
        kwargs: dict | None = None,
        *,
        base_url: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        kwargs = kwargs or {}
        self.model_name = model_name if model_name is not None else os.environ.get("LLM_MODEL_NAME", "")
        self.system_prompt = system_prompt
        self.prompt = prompt
        self.temperature = 1.0
        self.top_p = 1.0
        self.top_k = 1

        resolved_base = base_url or os.environ["LLM_BASE_URL"]
        resolved_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY", None)
        self.client = OpenAI(
            base_url=resolved_base,
            api_key=resolved_key,
        )

    def _build_messages(self, prompt_text: str, context: str = "") -> list:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})

        full_prompt = prompt_text + context
        messages.append({"role": "user", "content": [{"type": "text", "text": full_prompt}]})
        return messages

    def warmup(self, prompt: str | None = None) -> None:
        """Test that the server can answer correctly and the API key is valid."""
        self.generate(prompt=prompt or self.prompt, context="", max_new_tokens=5)

    def generate(
        self, prompt: str, context: str, max_new_tokens: int = 512, stop_token: str | None = "\n", **kwargs
    ) -> str:
        messages = self._build_messages(prompt or self.prompt, context)
        t_start_response_generation = time.monotonic()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=self.temperature,
            reasoning_effort="low",
            stop=[stop_token] if stop_token is not None else None,
            **kwargs,
        )

        logger.info(f"LLM response: {response}")
        text_response = response.choices[0].message.content
        if text_response:
            text_response = text_response.strip()
            text_response = text_response.split("\n")[0]
        else:
            logger.warning(
                f"LLM returned empty/None content (finish_reason={response.choices[0].finish_reason}), full response: {response}"
            )
            text_response = ""

        t_end_response_generation = time.monotonic()
        time_to_respond = t_end_response_generation - t_start_response_generation
        logger.info(f"LLM response generation took {time_to_respond:.2f} seconds")
        return text_response
