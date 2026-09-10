# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..inference_utils.retrieval_profiles import RetrievalProfile

from ..llm import (
    get_llm,
    load_reference_prompt_template,
    REFERENCE_PROMPT_TEMPLATE_FILE,
    REFERENCE_PROMPT_TEMPLATE_FILE_SIMPLIFIED,
    REFERENCE_SUMMARIZATION_PROMPT_TEMPLATE_FILE,
    LLMClient,
)
import logging

logger = logging.getLogger(__name__)


# History entry: (number of turns at the time the reference was generated, reference text)
ReferenceHistory = list[tuple[int, str]]


class LLMReferenceGenerator:
    """Generates reference text from conversation context using an LLM.

    This object is stateless across conversations: per-channel state (reference
    history) is supplied by the caller.
    """

    def __init__(
        self,
        summarization: bool = False,
        retrieval_profiles: list[RetrievalProfile] | None = None,
        *,
        prompt_style: str = "simplified",
    ):
        """
        Initialize reference manager

        Args:
            summarization: Whether to summarize the reference text
            retrieval_profiles: If at least two profiles, use per-profile endpoints; else env `get_llm`.
            prompt_style: For fewer than two profiles: bundled template. With multiple profiles, each profile's ``prompt_style`` is used instead.
        """
        self.summarization = summarization
        self.retrieval_profiles = list(retrieval_profiles or [])
        self._llm_by_id: dict[str, LLMClient] = {}

        if summarization:
            prompt_type = "reference_summarization"
        else:
            prompt_type = "reference"

        self._default_profile_id: str | None = None
        if len(self.retrieval_profiles) >= 2:
            from ..inference_utils.retrieval_profiles import default_profile_id

            for p in self.retrieval_profiles:
                if summarization:
                    template_file = REFERENCE_SUMMARIZATION_PROMPT_TEMPLATE_FILE
                else:
                    template_file = (
                        REFERENCE_PROMPT_TEMPLATE_FILE_SIMPLIFIED
                        if p.prompt_style == "simplified"
                        else REFERENCE_PROMPT_TEMPLATE_FILE
                    )
                prompt_text = load_reference_prompt_template(filename=template_file)
                self._llm_by_id[p.id] = LLMClient(
                    system_prompt="You are a helpful assistant.",
                    prompt=prompt_text,
                    base_url=p.base_url,
                    model_name=p.model,
                    api_key=p.api_key,
                )
            self._default_profile_id = default_profile_id(self.retrieval_profiles)
            self.llm = self._llm_by_id[self._default_profile_id]
        else:
            self.llm = get_llm(
                system_prompt="You are a helpful assistant.",
                prompt_type=prompt_type,
                prompt_style=prompt_style if prompt_type == "reference" else "original",
            )

        self._warmup_all_retrieval_llms()

    def _warmup_all_retrieval_llms(self) -> None:
        """Call ``warmup()`` on each retrieval ``LLMClient`` (separate endpoints). Single env LLM path warms ``self.llm`` only."""
        if self._llm_by_id:
            for profile_id, llm in self._llm_by_id.items():
                logger.info(
                    "[Reference] Warming up retrieval profile id=%r model=%r",
                    profile_id,
                    getattr(llm, "model_name", "?"),
                )
                llm.warmup()
        else:
            self.llm.warmup()

    def reference_model_display_name(self, active_profile_id: str | None = None) -> str:
        """Model name for the given (or default) retrieval profile."""
        if active_profile_id and active_profile_id in self._llm_by_id:
            return getattr(self._llm_by_id[active_profile_id], "model_name", "")
        return getattr(self.llm, "model_name", "")

    def process_reference_text(self, context: str, history: ReferenceHistory) -> tuple[str, int]:
        """
        Process reference text from conversation context

        Args:
            context: Conversation history context

        Returns:
            Processed reference text, number of turns before the reference
        """
        turns = []
        for turn in context.split("\n"):
            if turn.startswith("user:"):
                role = "Human"
                text = turn.split("user:")[1].strip()
                text = "".join([char for char in text if char.isprintable()]).strip()
                turns.append((role, text))
            elif turn.startswith("moshi:"):
                role = "moshi"
                text = turn.split("moshi:")[1].strip()
                text = "".join([char for char in text if char.isprintable()]).strip()
                turns.append((role, text))
        # The context may contain some initial part of the moshi turn that requires RAG, so we remove the last turn if it is from moshi
        if len(turns) > 0 and turns[-1][0] == "moshi":
            turns = turns[:-1]
        if len(turns) > 0 and turns[0][0] == "moshi":
            turns = turns[1:]

        # Format the context with the reference history
        processed_context = ""
        j = 0
        for i, (role, text) in enumerate(turns):
            if text:
                processed_context += f"{role}: {text}\n"
            else:
                processed_context += f"{role}:\n"
            if j < len(history) and i + 1 == history[j][0]:
                processed_context += f"Reference: {history[j][1]}\n"
                j += 1
        processed_context += "Reference:"
        return processed_context, len(turns)

    async def generate_reference_text(
        self,
        context: str,
        history: ReferenceHistory,
        active_profile_id: str | None = None,
        llm_call_timeout: float | None = None,
        max_tokens: int = 512,
    ) -> tuple[str, str, int, str]:
        """
        Generate reference text from conversation context.

        Args:
            context: Conversation history context
            history: Per-channel reference history
            active_profile_id: User-selected retrieval profile id (multi-profile mode)
            llm_call_timeout: Per-LLM ``generate`` deadline (seconds). ``None`` = no limit (e.g. warmup).
            max_tokens: Maximum number of new tokens to generate.

        Returns:
            ``(processed_context, reference_text, num_turns, source_model_display_name)`` where
            ``source_model_display_name`` is the model that produced ``reference_text``.
        """
        context, num_turns = self.process_reference_text(context, history)

        if num_turns == 0:
            logger.info("[Reference] No conversation turns yet, skipping LLM generation")
            return context, "", num_turns, ""

        loop = asyncio.get_event_loop()

        async def _run_one(llm: LLMClient) -> str:
            try:
                fut = loop.run_in_executor(None, llm.generate, llm.prompt, context, max_tokens)
                if llm_call_timeout is not None and llm_call_timeout > 0:
                    t = await asyncio.wait_for(fut, timeout=llm_call_timeout)
                else:
                    t = await fut
                return (t or "").strip()
            except asyncio.TimeoutError:
                logger.warning(
                    "[Reference] retrieval LLM %s timed out after %ss",
                    getattr(llm, "model_name", "?"),
                    llm_call_timeout,
                )
                return ""
            except Exception as e:
                logger.warning(
                    "[Reference] retrieval LLM %s failed: %s",
                    getattr(llm, "model_name", "?"),
                    e,
                )
                return ""

        lm_display = ""
        summarizer_llm = self.llm

        if len(self.retrieval_profiles) >= 2 and self._default_profile_id:
            active_llm = self._llm_by_id.get(active_profile_id) if active_profile_id else None
            default_llm = self._llm_by_id[self._default_profile_id]
            if active_llm is None:
                logger.warning("[Reference] unknown profile %r, falling back to default", active_profile_id)
                active_llm = default_llm
            if active_profile_id == self._default_profile_id:
                reference_text = await _run_one(active_llm)
                lm_display = getattr(active_llm, "model_name", "") if reference_text else ""
                summarizer_llm = active_llm
            else:
                primary_text, fallback_text = await asyncio.gather(
                    _run_one(active_llm),
                    _run_one(default_llm),
                )
                if primary_text:
                    reference_text = primary_text
                    lm_display = getattr(active_llm, "model_name", "")
                    summarizer_llm = active_llm
                else:
                    reference_text = fallback_text
                    lm_display = getattr(default_llm, "model_name", "") if reference_text else ""
                    summarizer_llm = default_llm if reference_text else active_llm
                    if reference_text:
                        logger.warning(
                            "[Reference] active profile %r returned empty; using default profile %r",
                            active_profile_id,
                            self._default_profile_id,
                        )
        else:
            try:
                fut = loop.run_in_executor(None, self.llm.generate, self.llm.prompt, context, max_tokens)
                if llm_call_timeout is not None and llm_call_timeout > 0:
                    reference_text = await asyncio.wait_for(fut, timeout=llm_call_timeout)
                else:
                    reference_text = await fut
                reference_text = (reference_text or "").strip()
            except asyncio.TimeoutError:
                logger.warning(
                    "[Reference] retrieval LLM %s timed out after %ss",
                    getattr(self.llm, "model_name", "?"),
                    llm_call_timeout,
                )
                reference_text = ""
            lm_display = getattr(self.llm, "model_name", "") if reference_text else ""
            summarizer_llm = self.llm

        if self.summarization and reference_text:
            reference_text = await loop.run_in_executor(
                None,
                summarizer_llm.generate,
                summarizer_llm.prompt,
                f"{context}{reference_text}\nSummarized reference:",
                max_tokens,
            )
            lm_display = getattr(summarizer_llm, "model_name", lm_display)

        return context, reference_text, num_turns, lm_display

    def warmup(self):
        """Warm up every retrieval profile endpoint, then run one full async reference pass (prompt/history path)."""
        self._warmup_all_retrieval_llms()
        asyncio.run(self.generate_reference_text("", [], llm_call_timeout=None))
