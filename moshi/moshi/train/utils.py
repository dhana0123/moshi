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

    def start_step(self):
        self.step += 1
        self.begin_step_time = time.time()

    def end_step(self, n_batch_tokens: int):
        self.this_step_time = time.time() - self.begin_step_time
        self.this_step_tokens = n_batch_tokens
        self.elapsed_time += self.this_step_time
        self.n_seen_tokens += n_batch_tokens

    @property
    def wps(self):
        if self.this_step_time <= 0:
            return 0.0
        return self.this_step_tokens / self.this_step_time


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
