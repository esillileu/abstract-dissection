"""Small helpers for executor-owned runtime profiling summaries."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping

from mlprosection.profiling import RuntimeMonitor, profiling_config_from_mapping
from mlprosection.profiling.backend import create_backend_profiler


def create_runtime_monitor(backend, profiling: Mapping[str, object]) -> RuntimeMonitor:
    """Create a monitor after validating the RunSpec profiling section."""

    profiling_config_from_mapping(profiling)
    return RuntimeMonitor(create_backend_profiler(backend))


@contextmanager
def training_summary(monitor: RuntimeMonitor) -> Iterator[None]:
    """Collect run boundary memory and total train wall time."""

    monitor.snapshot_memory("run.start")
    monitor.update_memory_peaks()
    try:
        with monitor.timer("train_total"):
            yield
    finally:
        monitor.snapshot_memory("run.end")
        monitor.update_memory_peaks()
