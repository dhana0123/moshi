# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Handling speaker transitions, text buffering, and context management for text display and RAG reference text generation."""

from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


ROLE_PREFIX_MAPPING = {
    "model": "moshi: ",
    "user": "user: ",
}


@dataclass
class TurnManager:
    """Manages VAD state, speaker transitions, text buffering, and context."""

    # VAD parameters
    window_size: int = 4
    threshold: float = 0.5
    warmup_steps: int = 20
    vad_history: list[float] = field(default_factory=list)
    _step_count: int = 0

    # Speaker transition parameters
    stt_wait_steps: int = 0  # Number of steps to wait before switching to model
    _wait_counter: int = 0  # Counter for wait steps
    _pending_speaker: str | None = None  # Track pending speaker switch

    # Conversation/state tracking
    init_active_speaker: str = "model"
    active_speaker: str = "model"
    conversation_context: str = ""
    model_text_buffer: list[tuple[str, str]] = field(default_factory=list)
    user_text_buffer: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self):
        # Little trick to make the initial speaker properly displayed
        if self.init_active_speaker == "user":
            self.user_text_buffer = [("user: ", "user")]
        elif self.init_active_speaker == "model":
            self.model_text_buffer = [("moshi: ", "model")]
        self.active_speaker = self.init_active_speaker

    def get_context(self) -> str:
        """Return accumulated conversation context."""
        return self.conversation_context

    def update_vad(self, vad_value: float):
        """Update VAD history with the latest measurement."""
        self._step_count += 1
        self.vad_history.append(vad_value)
        if len(self.vad_history) > self.window_size:
            self.vad_history.pop(0)

    def _handle_pending_speaker_switch(self) -> str | None:
        """Handle counting and pending speaker switch."""
        if self._pending_speaker is None:
            return None
        if self._wait_counter == 0:
            speaker = self._pending_speaker
            self._pending_speaker = None
            return speaker
        self._wait_counter -= 1
        return None

    def _update_active_speaker(self) -> str:
        """Determine new active speaker based on VAD history"""
        current_speaker = self.active_speaker
        if len(self.vad_history) < self.window_size:
            return current_speaker

        vad_neg = all([value > self.threshold for value in self.vad_history])
        vad_pos = sum(self.vad_history) / len(self.vad_history) < self.threshold
        warmup_done = self._step_count >= self.warmup_steps

        if warmup_done and vad_pos and current_speaker == "model":
            if self._pending_speaker != "user":
                self._pending_speaker = "user"
                self._wait_counter = 0
                logger.info(
                    f"[VAD] User started speaking (history={[f'{v:.3f}' for v in self.vad_history]})",
                )
        elif vad_neg and current_speaker == "user":
            if self.model_text_buffer:
                if self._pending_speaker != "model":
                    self._pending_speaker = "model"
                    # After model starts speaking, wait for the specified number of steps before switching speaker
                    # This is to ensure that the stt model has enough time to generate the full user transcript
                    self._wait_counter = self.stt_wait_steps
                    logger.info(
                        f"[VAD] User stopped speaking (history={[f'{v:.3f}' for v in self.vad_history]}), waiting",
                    )
            else:
                logger.info("[VAD] User stopped speaking but LM buffer empty, remaining in user turn")

        switch_speaker = self._handle_pending_speaker_switch()
        if switch_speaker is not None:
            logger.info(f"[VAD] Waiting ended, switching to {switch_speaker}")
            return switch_speaker
        return current_speaker

    def _handle_state_transition(self, new_active_speaker: str) -> list[tuple[str, str]]:
        """Handle state transitions between speakers and flush buffers."""
        outputs: list[tuple[str, str]] = []

        if self.active_speaker != new_active_speaker:
            logger.info(f"[State] Switching to {new_active_speaker}")
            # Add the new active speaker to the outputs and update the active speaker
            outputs.append(("\n" + ROLE_PREFIX_MAPPING.get(new_active_speaker, ""), new_active_speaker))
            self.active_speaker = new_active_speaker

        if new_active_speaker == "user":
            if self.user_text_buffer:
                outputs.extend(self.user_text_buffer)
                buffered_text = "".join([item[0] for item in self.user_text_buffer])
                logger.info(f"[Flushed STT buffer] {buffered_text}")
                self.user_text_buffer = []
        else:
            if self.model_text_buffer:
                outputs.extend(self.model_text_buffer)
                buffered_text = "".join([item[0] for item in self.model_text_buffer])
                logger.info(f"[Flushed LM buffer] {buffered_text}")
                self.model_text_buffer = []

        return outputs

    def handle_spoken_text(self, model_text: str | None = None, user_text: str | None = None) -> list[tuple[str, str]]:
        """Process spoken text for both model and user and decide whether to display or buffer it."""
        new_active_speaker = self._update_active_speaker()
        outputs: list[tuple[str, str]] = self._handle_state_transition(new_active_speaker)

        if model_text is not None:
            if self.active_speaker == "model":
                outputs.append((model_text, "model"))
                logger.info(f"[Display Model] '{model_text.strip()}'")
            else:
                if not self.model_text_buffer:
                    logger.info(
                        f"[Buffer] buffering model text while user is active. first_chunk='{model_text.strip()}'"
                    )
                self.model_text_buffer.append((model_text, "model"))
                logger.info(f"[Buffered Model] '{model_text.strip()}'")
        if user_text is not None:
            if self.active_speaker == "user":
                outputs.append((user_text, "user"))
                logger.info(f"[Display User] '{user_text.strip()}'")
            else:
                if not self.user_text_buffer:
                    logger.info(
                        f"[Buffer] buffering user text while model is active. first_chunk='{user_text.strip()}'"
                    )
                self.user_text_buffer.append((user_text, "user"))
                logger.info(f"[Buffered User] '{user_text.strip()}'")

        self._update_context("".join([item[0] for item in outputs]))

        return outputs

    def _update_context(self, text: str):
        self.conversation_context += text.replace("[RAG]", "")

    def reset(self):
        """Reset VAD state and conversation buffers."""
        self.vad_history = []
        self._step_count = 0
        self._wait_counter = 0
        self._pending_speaker = None
        self.active_speaker = self.init_active_speaker
        self.conversation_context = ""
        self.model_text_buffer = []
        self.user_text_buffer = []
        self.__post_init__()
