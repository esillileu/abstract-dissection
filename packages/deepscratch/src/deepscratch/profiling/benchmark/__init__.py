"""Reusable synchronized microbenchmark and runtime-estimation primitives."""

from __future__ import annotations

from .estimate import estimate_training_time
from .recorder import SectionRecorder
from .results import (
    BenchmarkResult,
    Operation,
    Prepare,
    TimingStats,
    TrainingTimeEstimate,
    UpdateBenchmarkResult,
)
from .runner import BenchmarkRunner

__all__ = [
    "BenchmarkResult",
    "BenchmarkRunner",
    "Operation",
    "Prepare",
    "SectionRecorder",
    "TimingStats",
    "TrainingTimeEstimate",
    "UpdateBenchmarkResult",
    "estimate_training_time",
]
