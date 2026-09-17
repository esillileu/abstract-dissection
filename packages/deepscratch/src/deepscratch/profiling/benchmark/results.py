"""Result containers and timing statistics for microbenchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from statistics import mean, stdev

from deepscratch.profiling.utils import summarize_values

Operation = Callable[[], object]
Prepare = Callable[[], object]


@dataclass(frozen=True)
class TimingStats:
    count: int
    mean_ms: float
    stdev_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float

    @classmethod
    def from_values(cls, values: list[float]) -> TimingStats:
        if not values:
            raise ValueError("at least one timing sample is required")
        summary = summarize_values(values)
        return cls(
            count=len(values),
            mean_ms=mean(values),
            stdev_ms=stdev(values) if len(values) > 1 else 0.0,
            min_ms=min(values),
            max_ms=max(values),
            p50_ms=float(summary["p50"]),
            p95_ms=float(summary["p95"]),
        )


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    warmup_iterations: int
    measured_iterations: int
    repetitions: int
    warmup_total_ms: float
    warmup_mean_ms: float
    timing: TimingStats

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class UpdateBenchmarkResult:
    """Cold, individual steady-state, and continuous-window update timings."""

    name: str
    cold_ms: float
    warmup_iterations: int
    measured_iterations: int
    repetitions: int
    warmup_total_ms: float
    warmup_mean_ms: float
    event_timing: TimingStats
    timing: TimingStats

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingTimeEstimate:
    updates_per_epoch: int
    epochs: int
    mean_seconds_per_epoch: float
    repeat_stdev_seconds_per_epoch: float
    mean_seconds_total: float
    repeat_stdev_seconds_total: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = [
    "BenchmarkResult",
    "Operation",
    "Prepare",
    "TimingStats",
    "TrainingTimeEstimate",
    "UpdateBenchmarkResult",
]
