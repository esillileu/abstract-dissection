"""Reusable synchronized microbenchmark and runtime-estimation primitives."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from statistics import mean, stdev
from time import perf_counter
from typing import Callable, Iterator

from .utils import summarize_values


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


class BenchmarkRunner:
    """Measure operations with backend synchronization at timing boundaries."""

    def __init__(self, backend) -> None:
        self.backend = backend

    def measure_iterations(
        self,
        name: str,
        operation: Operation,
        *,
        warmup_iterations: int,
        measured_iterations: int,
        prepare: Prepare | None = None,
    ) -> BenchmarkResult:
        """Measure individual calls, excluding optional per-call preparation."""
        self._validate(warmup_iterations, measured_iterations, 1)
        warmup_values = [
            self._measure_once(name, operation, prepare=prepare)
            for _ in range(warmup_iterations)
        ]
        values = [
            self._measure_once(name, operation, prepare=prepare)
            for _ in range(measured_iterations)
        ]
        return BenchmarkResult(
            name=name,
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
            repetitions=measured_iterations,
            warmup_total_ms=sum(warmup_values),
            warmup_mean_ms=mean(warmup_values) if warmup_values else 0.0,
            timing=TimingStats.from_values(values),
        )

    def measure_windows(
        self,
        name: str,
        operation: Operation,
        *,
        warmup_iterations: int,
        iterations_per_window: int,
        repetitions: int,
    ) -> BenchmarkResult:
        """Measure repeated operation windows and normalize to one iteration."""
        self._validate(warmup_iterations, iterations_per_window, repetitions)
        warmup_total = (
            self._measure_window(
                f"{name}.warmup",
                operation,
                warmup_iterations,
            )
            if warmup_iterations
            else 0.0
        )
        values = [
            self._measure_window(name, operation, iterations_per_window)
            / iterations_per_window
            for _ in range(repetitions)
        ]
        return BenchmarkResult(
            name=name,
            warmup_iterations=warmup_iterations,
            measured_iterations=iterations_per_window,
            repetitions=repetitions,
            warmup_total_ms=warmup_total,
            warmup_mean_ms=(
                warmup_total / warmup_iterations if warmup_iterations else 0.0
            ),
            timing=TimingStats.from_values(values),
        )

    def measure_update_protocol(
        self,
        name: str,
        operation: Operation,
        *,
        warmup_iterations: int,
        measured_iterations: int,
        repetitions: int,
    ) -> UpdateBenchmarkResult:
        """Separate one cold update from steady event and throughput timings.

        CUDA event pairs are recorded around consecutive updates and resolved
        after one synchronization.  The independent continuous windows are the
        authoritative throughput measurement used for extrapolation.
        """
        self._validate(warmup_iterations, measured_iterations, repetitions)
        cold_ms = self._measure_window(f"{name}.cold", operation, 1)
        warmup_total = (
            self._measure_window(
                f"{name}.warmup",
                operation,
                warmup_iterations,
            )
            if warmup_iterations
            else 0.0
        )
        event_values = self._measure_consecutive_updates(
            f"{name}.steady_events",
            operation,
            measured_iterations,
        )
        window_values = [
            self._measure_window(name, operation, measured_iterations)
            / measured_iterations
            for _ in range(repetitions)
        ]
        return UpdateBenchmarkResult(
            name=name,
            cold_ms=cold_ms,
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
            repetitions=repetitions,
            warmup_total_ms=warmup_total,
            warmup_mean_ms=(
                warmup_total / warmup_iterations if warmup_iterations else 0.0
            ),
            event_timing=TimingStats.from_values(event_values),
            timing=TimingStats.from_values(window_values),
        )

    def _measure_consecutive_updates(
        self,
        name: str,
        operation: Operation,
        iterations: int,
    ) -> list[float]:
        if not self.backend.is_gpu:
            values = []
            with self.backend.range(name):
                for _ in range(iterations):
                    started = perf_counter()
                    operation()
                    values.append((perf_counter() - started) * 1_000)
            return values

        xp = self.backend.xp
        event_pairs = [
            (xp.cuda.Event(), xp.cuda.Event()) for _ in range(iterations)
        ]
        self.backend.synchronize()
        with self.backend.range(name):
            for start, end in event_pairs:
                start.record()
                operation()
                end.record()
        self.backend.synchronize()
        return [
            float(xp.cuda.get_elapsed_time(start, end))
            for start, end in event_pairs
        ]

    def _measure_once(
        self,
        name: str,
        operation: Operation,
        *,
        prepare: Prepare | None,
    ) -> float:
        if prepare is not None:
            prepare()
        return self._measure_window(name, operation, 1)

    def _measure_window(
        self,
        name: str,
        operation: Operation,
        iterations: int,
    ) -> float:
        self.backend.synchronize()
        started = perf_counter()
        with self.backend.range(name):
            for _ in range(iterations):
                operation()
        self.backend.synchronize()
        return (perf_counter() - started) * 1_000

    @staticmethod
    def _validate(
        warmup_iterations: int,
        measured_iterations: int,
        repetitions: int,
    ) -> None:
        if warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        if measured_iterations < 1:
            raise ValueError("measured_iterations must be positive")
        if repetitions < 1:
            raise ValueError("repetitions must be positive")


class SectionRecorder:
    """Record explicitly marked model or operation sections."""

    def __init__(self, backend) -> None:
        self.backend = backend
        self.values_ms: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        self.backend.synchronize()
        started = perf_counter()
        with self.backend.range(name):
            yield
        self.backend.synchronize()
        self.values_ms[name].append((perf_counter() - started) * 1_000)

    def stats(self) -> dict[str, TimingStats]:
        return {
            name: TimingStats.from_values(values)
            for name, values in self.values_ms.items()
        }


def estimate_training_time(
    update_timing: TimingStats,
    *,
    dataset_samples: int,
    batch_size: int,
    epochs: int,
    drop_last: bool = True,
    cold_update_ms: float | None = None,
) -> TrainingTimeEstimate:
    """Extrapolate update timing, optionally charging one cold update once."""
    if min(dataset_samples, batch_size, epochs) < 1:
        raise ValueError("dataset_samples, batch_size, and epochs must be positive")
    updates_per_epoch = (
        dataset_samples // batch_size
        if drop_last
        else (dataset_samples + batch_size - 1) // batch_size
    )
    total_updates = updates_per_epoch * epochs
    if cold_update_ms is None:
        steady_updates = total_updates
        mean_total_ms = update_timing.mean_ms * steady_updates
    else:
        if cold_update_ms < 0:
            raise ValueError("cold_update_ms must be non-negative")
        steady_updates = max(total_updates - 1, 0)
        mean_total_ms = cold_update_ms + update_timing.mean_ms * steady_updates
    repeat_stdev_total_ms = update_timing.stdev_ms * steady_updates
    return TrainingTimeEstimate(
        updates_per_epoch=updates_per_epoch,
        epochs=epochs,
        mean_seconds_per_epoch=mean_total_ms / epochs / 1_000,
        repeat_stdev_seconds_per_epoch=(
            repeat_stdev_total_ms / epochs / 1_000
        ),
        mean_seconds_total=mean_total_ms / 1_000,
        repeat_stdev_seconds_total=repeat_stdev_total_ms / 1_000,
    )
