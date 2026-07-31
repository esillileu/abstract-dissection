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
def training_summary(
    monitor: RuntimeMonitor,
    *,
    synchronize: bool = False,
) -> Iterator[None]:
    """Collect run boundary memory and total train wall time.

    ``synchronize`` adds a second, explicitly synchronized timer around the
    complete training call.  It synchronizes only at the two run boundaries,
    rather than perturbing execution at every probe window.
    """

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
