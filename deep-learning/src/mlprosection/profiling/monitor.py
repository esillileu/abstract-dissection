from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter_ns
from typing import Iterator

from .backend import BackendProfiler, MetricValue
from .utils import summarize_values


class RuntimeMonitor:
    """Collects low-overhead runtime timings and memory metrics."""

    def __init__(self, backend_profiler: BackendProfiler) -> None:
        self.backend_profiler = backend_profiler
        self.timings_ms: dict[str, list[float]] = {}
        self.memory_snapshots: dict[str, dict[str, MetricValue]] = {}
        self.memory_peaks: dict[str, MetricValue] = {}
        self.scalar_metrics: dict[str, MetricValue] = {}

    @contextmanager
    def timer(
        self,
        name: str,
        *,
        synchronize: bool = False,
    ) -> Iterator[None]:
        if synchronize:
            self.backend_profiler.synchronize()
        start = perf_counter_ns()
        try:
            yield
        finally:
            if synchronize:
                self.backend_profiler.synchronize()
            elapsed_ms = (perf_counter_ns() - start) / 1_000_000
            self.timings_ms.setdefault(name, []).append(elapsed_ms)

    def snapshot_memory(self, name: str, *, synchronize: bool = False) -> None:
        if synchronize:
            self.backend_profiler.synchronize()

        self.memory_snapshots[name] = self.backend_profiler.memory_stats()

    def update_memory_peaks(self) -> None:
        for key, value in self.backend_profiler.memory_stats().items():
            current = self.memory_peaks.get(key)
            if current is None or value > current:
                self.memory_peaks[key] = value

    def set_metric(self, name: str, value: MetricValue) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"metric value must be int or float: {name}")
        self.scalar_metrics[name] = value

    def metrics(self) -> dict[str, MetricValue]:
        metrics: dict[str, MetricValue] = dict(self.scalar_metrics)

        for name, values in self.timings_ms.items():
            for key, value in summarize_values(values).items():
                metric_name = f"runtime.{name}.{key}"
                if key != "count":
                    metric_name = f"{metric_name}_ms"
                metrics[metric_name] = value

        for name, snapshot in self.memory_snapshots.items():
            for key, value in snapshot.items():
                metrics[f"memory.{name}.{key}"] = value

        for key, value in self.memory_peaks.items():
            metrics[f"memory.peak_sampled.{key}"] = value

        return metrics
