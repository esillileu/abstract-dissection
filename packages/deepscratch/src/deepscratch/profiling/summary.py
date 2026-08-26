"""Helpers for executor-owned runtime profiling summaries."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from .backend import create_backend_profiler
from .config import profiling_config_from_mapping
from .monitor import RuntimeMonitor


def create_runtime_monitor(backend, profiling: Mapping[str, object]) -> RuntimeMonitor:
    """Create a monitor after validating the RunSpec profiling section."""
    profiling_config_from_mapping(profiling)
    return RuntimeMonitor(create_backend_profiler(backend))


@contextmanager
def training_summary(
    monitor: RuntimeMonitor,
    *,
    synchronize: bool = False,
) -> Iterator[None]:
    """Collect run boundary memory and total train wall time."""
    monitor.snapshot_memory("run.start")
    monitor.update_memory_peaks()
    try:
        with monitor.timer("train_total"):
            if synchronize:
                with monitor.timer("train_synchronized", synchronize=True):
                    yield
            else:
                yield
    finally:
        monitor.snapshot_memory("run.end")
        monitor.update_memory_peaks()
