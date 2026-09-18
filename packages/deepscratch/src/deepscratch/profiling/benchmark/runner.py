"""Synchronized benchmark execution runner."""

from __future__ import annotations

from statistics import mean
from time import perf_counter

from .results import (
    BenchmarkResult,
    Operation,
    Prepare,
    TimingStats,
    UpdateBenchmarkResult,
)


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
        event_pairs = [(xp.cuda.Event(), xp.cuda.Event()) for _ in range(iterations)]
        self.backend.synchronize()
        with self.backend.range(name):
            for start, end in event_pairs:
                start.record()
                operation()
                end.record()
        self.backend.synchronize()
        return [
            float(xp.cuda.get_elapsed_time(start, end)) for start, end in event_pairs
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


__all__ = ["BenchmarkRunner"]
