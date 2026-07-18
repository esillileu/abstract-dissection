from __future__ import annotations

import os
from typing import Protocol

import psutil


MetricValue = int | float


class BackendProfiler(Protocol):
    """Backend-specific synchronization and memory adapter."""

    @property
    def name(self) -> str:
        ...

    def synchronize(self) -> None:
        ...

    def memory_stats(self) -> dict[str, MetricValue]:
        ...


class NumPyBackendProfiler:
    """Profiler adapter for NumPy-backed execution."""

    name = "numpy"

    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())

    def synchronize(self) -> None:
        return None

    def memory_stats(self) -> dict[str, MetricValue]:
        info = self.process.memory_info()
        stats: dict[str, MetricValue] = {
            "cpu.rss_bytes": info.rss,
            "cpu.vms_bytes": info.vms,
        }

        try:
            full_info = self.process.memory_full_info()
        except (psutil.Error, AttributeError):
            return stats

        for name in ("uss", "pss", "swap"):
            value = getattr(full_info, name, None)
            if value is not None:
                stats[f"cpu.{name}_bytes"] = value

        return stats


class CuPyBackendProfiler:
    """Profiler adapter for CuPy-backed execution."""

    name = "cupy"

    def __init__(self, cupy_module) -> None:
        self.cp = cupy_module
        self.process = psutil.Process(os.getpid())
        self.memory_pool = self.cp.get_default_memory_pool()
        self.pinned_memory_pool = self.cp.get_default_pinned_memory_pool()

    def synchronize(self) -> None:
        self.cp.cuda.get_current_stream().synchronize()

    def memory_stats(self) -> dict[str, MetricValue]:
        stats = NumPyBackendProfiler.memory_stats(self)
        stats.update(
            {
                "gpu.pool_used_bytes": self.memory_pool.used_bytes(),
                "gpu.pool_reserved_bytes": self.memory_pool.total_bytes(),
                "gpu.pinned_free_blocks": self.pinned_memory_pool.n_free_blocks(),
            }
        )
        return stats


def create_backend_profiler(backend) -> BackendProfiler:
    """Create a profiler adapter for the project's Backend object."""

    backend_name = getattr(backend, "name", None)
    if backend_name is None:
        backend_name = getattr(getattr(backend, "xp", None), "__name__", None)

    if backend_name == "numpy":
        return NumPyBackendProfiler()

    if backend_name == "cupy":
        return CuPyBackendProfiler(backend.xp)

    raise ValueError(f"Unsupported backend for profiling: {backend_name}")
