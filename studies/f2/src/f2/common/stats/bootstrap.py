"""Resampling bootstrap variance engine and confidence interval calculation."""

from __future__ import annotations

import math
import random
from typing import Sequence


def bootstrap_confidence_interval(
    estimates: Sequence[float], alpha: float = 0.05
) -> tuple[float, float]:
    """Compute empirical percentile confidence interval from bootstrap replicate estimates."""
    if not estimates:
        return 0.0, 0.0
    sorted_vals = sorted(estimates)
    n = len(sorted_vals)
    low_idx = int(math.floor(n * (alpha / 2.0)))
    high_idx = int(math.ceil(n * (1.0 - alpha / 2.0))) - 1
    return sorted_vals[max(0, low_idx)], sorted_vals[min(n - 1, high_idx)]


class BootstrapVarianceEngine:
    """Multi-stage survey and residual bootstrap engine."""

    def __init__(self, reps: int = 1000, seed: int = 42) -> None:
        self.reps = reps
        self.seed = seed
        self.rng = random.Random(seed)

    def draw_gamma_multipliers(self, n: int, scale: float = 1.0) -> list[float]:
        """Draw Gamma(1, 1) standard exponential perturbation multipliers."""
        return [self.rng.gammavariate(1.0, 1.0) * scale for _ in range(n)]


__all__ = [
    "BootstrapVarianceEngine",
    "bootstrap_confidence_interval",
]
