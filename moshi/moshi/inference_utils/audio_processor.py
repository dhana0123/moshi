# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


class AudioProcessor:
    """Handles audio processing operations."""

    def __init__(self, power_threshold: float | None = None):
        """Initialize audio processor.

        Args:
            power_threshold: RMS power threshold in dB. If None, no filtering is applied.
        """
        self.power_threshold = power_threshold

    def filter_by_power(self, chunk: torch.Tensor) -> torch.Tensor:
        """Zero-out chunk if RMS power (in dB) is below threshold.

        Args:
            chunk: Tensor [B=1, C=1, T]
        Returns:
            Possibly modified chunk with silence applied if below threshold.
        """
        if self.power_threshold is None:
            return chunk
        # RMS over time axis
        rms = torch.sqrt(torch.mean(chunk**2, dim=-1, keepdim=True))  # [1,1,1]
        db = 10 * torch.log10(rms**2 + 1e-16)
        if (db < self.power_threshold).item():
            return torch.zeros_like(chunk)
        return chunk
