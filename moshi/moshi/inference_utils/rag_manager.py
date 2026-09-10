# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""RAG (Retrieval Augmented Generation) manager for reference text generation.

The manager owns per-channel reference history and pending generation task
lifecycle. Use it as an async context manager so pending background tasks are
cancelled and awaited on exit.
"""

import asyncio
import contextlib
import time
from typing import Awaitable, Callable
import logging
from ..reference import LLMReferenceGenerator
from ..reference.llm_reference_generator import ReferenceHistory

logger = logging.getLogger(__name__)


class RAGManager:
    """Manages asynchronous RAG reference generation scoped to one channel."""

    def __init__(
        self,
        reference_generator: LLMReferenceGenerator,
        rag_timeout: float = 1.5,
        max_tokens: int = 512,
        gt_reference_text: str | None = None,
    ):
        self.reference_generator = reference_generator
        self.rag_timeout = rag_timeout
        self.max_tokens = max_tokens
        self.gt_reference_text = gt_reference_text
        self._history: ReferenceHistory = []
        self._wait_steps_remaining: int = 0
        self._wait_event: asyncio.Event | None = None
        self._pending_task: asyncio.Task | None = None
        self._stack: contextlib.AsyncExitStack | None = None
        self._active_profile_id: str | None = None

    async def __aenter__(self) -> "RAGManager":
        self._stack = contextlib.AsyncExitStack()
        await self._stack.__aenter__()
        self._stack.push_async_callback(self._cancel_and_await_pending)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        assert self._stack is not None
        try:
            return await self._stack.__aexit__(exc_type, exc, tb)
        finally:
            self._stack = None

    async def _cancel_and_await_pending(self):
        task = self._pending_task
        self._pending_task = None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def get_reference_text(self, context: str) -> tuple[str, str, float, str]:
        """Returns (query_context, reference_text, elapsed_seconds, lm_display_name)."""
        try:
            logger.info("[Reference] Generating reference")

            if self.gt_reference_text is not None:
                logger.info(f"[Reference] Using ground truth reference text: {self.gt_reference_text}")
                return "", self.gt_reference_text, 0.0, "Ground truth"

            retrieval_start_time = time.time()
            query, reference_text, num_turns, lm_label = await self.reference_generator.generate_reference_text(
                context,
                self._history,
                active_profile_id=self._active_profile_id,
                llm_call_timeout=self.rag_timeout,
                max_tokens=self.max_tokens,
            )
            retrieval_elapsed = time.time() - retrieval_start_time
            if num_turns > 0 and reference_text:
                self._history.append((num_turns, reference_text))
            logger.info(
                f"[Reference] Generated reference in {retrieval_elapsed:.3f}s: {reference_text}",
            )
            return query, reference_text, retrieval_elapsed, lm_label
        except asyncio.TimeoutError:
            logger.warning(
                f"[Reference] Reference generation timed out after {self.rag_timeout}s, returning empty string"
            )
            return "", "", self.rag_timeout, ""
        except Exception as e:
            logger.error(f"[Reference] Error generating reference: {e}, returning empty string")
            return "", "", self.rag_timeout, ""

    def warmup(self):
        """Warmup reference generator."""
        self.reference_generator.warmup()

    def set_retrieval_profile_id(self, profile_id: str) -> None:
        self._active_profile_id = profile_id

    async def trigger(
        self,
        task_group: asyncio.TaskGroup,
        wait_steps: int = 0,
        handle_reference_fn: Callable[..., Awaitable[None]] | None = None,
        context_provider: Callable[[], str] | None = None,
    ):
        """Trigger reference text generation in background."""
        if self._stack is None:
            raise RuntimeError("RAGManager.trigger called outside of `async with` scope")

        await self._cancel_and_await_pending()

        if wait_steps > 0:
            self._wait_steps_remaining = wait_steps
            self._wait_event = asyncio.Event()
            logger.info(f"[Reference] Started waiting for {wait_steps} steps")
        else:
            self._wait_event = None
            self._wait_steps_remaining = 0

        self._pending_task = task_group.create_task(self._background_task(handle_reference_fn, context_provider))

    async def _background_task(
        self,
        handle_reference_fn: Callable[..., Awaitable[None]] | None,
        context_provider: Callable[[], str] | None,
    ):
        logger.info("[Reference] Started new reference generation task in background")
        try:
            if self._wait_event is not None:
                await self._wait_event.wait()
                self._wait_event = None
            logger.info("[Reference] Waiting ended (including zero wait_steps)")
            if context_provider is not None:
                context = context_provider()
            else:
                context = ""
                logger.warning("[Reference] No context provider supplied, generating reference with empty context")
            logger.info(
                f"[Reference] Triggering retrieval with context_len={len(context)} snippet='...{context[-200:]}'"
            )
            _, reference_text, _, lm_label = await self.get_reference_text(context)
            if handle_reference_fn is not None:
                await handle_reference_fn(reference_text, lm_label)
            logger.info("[Reference] Background reference generation task completed")
        except asyncio.CancelledError:
            logger.info("[Reference] Reference generation cancelled")
            raise
        except Exception as e:
            logger.error(f"[Reference] Error generating reference: {e}")

    def step(self):
        """Signal that a step has passed. Used for step-based waiting."""
        if self._wait_steps_remaining > 0:
            self._wait_steps_remaining -= 1
            if self._wait_steps_remaining == 0 and self._wait_event is not None:
                self._wait_event.set()

    def cancel_pending(self):
        """Cancel any pending reference generation task."""
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()

    def reset(self, gt_reference_text: str | None = None):
        """Reset RAG manager state."""
        self.cancel_pending()
        self._wait_steps_remaining = 0
        self._wait_event = None
        self.gt_reference_text = gt_reference_text
        self._history = []
