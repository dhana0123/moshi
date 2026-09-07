# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import dataclasses
import logging
import time
from typing import Protocol

import torch

logger = logging.getLogger("moshi.train")


def format_eta(seconds: float) -> str:
    """Format seconds as a short human-readable duration (e.g. 2h15m, 45m12s)."""
    if seconds < 0 or seconds != seconds:  # NaN
        return "n/a"
    seconds = int(round(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d{hours:02d}h{minutes:02d}m"
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


@dataclasses.dataclass
class TrainState:
    max_steps: int
    step: int = 0
    elapsed_time: float = 0.0
    n_seen_tokens: int = 0
    this_step_time: float = 0.0
    begin_step_time: float = 0.0
    this_step_tokens: int = 0
    this_eval_perplexity: float | None = None
    this_eval_loss: float | None = None
    this_audio_loss: float | None = None
    this_text_loss: float | None = None
    # Per-step wall times (seconds), appended in end_step for ETA / benches.
    step_times: list[float] = dataclasses.field(default_factory=list)

    def start_step(self):
        self.step += 1
        self.begin_step_time = time.time()

    def end_step(self, n_batch_tokens: int):
        self.this_step_time = time.time() - self.begin_step_time
        self.this_step_tokens = n_batch_tokens
        self.elapsed_time += self.this_step_time
        self.n_seen_tokens += n_batch_tokens
        self.step_times.append(self.this_step_time)

    @property
    def wps(self):
        if self.this_step_time <= 0:
            return 0.0
        return self.this_step_tokens / self.this_step_time

    def rolling_step_time(self, window: int = 50) -> float | None:
        """Mean of the last ``window`` step times, or None if no steps yet."""
        if not self.step_times:
            return None
        recent = self.step_times[-window:]
        return sum(recent) / len(recent)

    def eta_seconds(self, window: int = 50) -> float | None:
        avg = self.rolling_step_time(window=window)
        if avg is None:
            return None
        remaining = max(self.max_steps - self.step, 0)
        return remaining * avg


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class Closable(Protocol):
    def close(self):
        pass


@contextlib.contextmanager
def logged_closing(thing: Closable, name: str):
    try:
        yield
    finally:
        logger.info("Closing: %s", name)
        thing.close()
