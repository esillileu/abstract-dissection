"""Statistics over normalized metric values."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import sqrt
from statistics import fmean, stdev


@dataclass(frozen=True)
class SeriesStatistics:
    count: int
    mean: float
    sample_standard_deviation: float
    standard_error: float
    minimum: float
    maximum: float


def summarize_series(values: Iterable[float]) -> SeriesStatistics:
    samples = tuple(float(value) for value in values)
    if not samples:
        raise ValueError("cannot summarize an empty series")
    deviation = stdev(samples) if len(samples) > 1 else 0.0
    return SeriesStatistics(
        count=len(samples),
        mean=fmean(samples),
        sample_standard_deviation=deviation,
        standard_error=deviation / sqrt(len(samples)),
        minimum=min(samples),
        maximum=max(samples),
    )
