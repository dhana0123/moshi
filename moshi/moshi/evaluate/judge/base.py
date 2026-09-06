# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Judge(ABC):
    """Abstract interface for duplex judges."""

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Run the judge on the given data."""
        pass
