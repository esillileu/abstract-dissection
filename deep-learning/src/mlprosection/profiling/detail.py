from __future__ import annotations

import cProfile
import logging
import pstats
import tracemalloc
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator

from .backend import BackendProfiler
from .config import ProfilingConfig
from .monitor import RuntimeMonitor

logger = logging.getLogger(__name__)


class DetailProfiler:
    """Manages optional step profiling, cProfile, and tracemalloc."""

    def __init__(
        self,
        config: ProfilingConfig,
        backend_profiler: BackendProfiler,
        monitor: RuntimeMonitor,
    ) -> None:
        self.config = config
        self.backend_profiler = backend_profiler
        self.monitor = monitor
        self._profile: cProfile.Profile | None = None
        self._tracemalloc_started_here = False
        self._python_profile_enabled = False
        self._memory_profile_enabled = False
        self._gpu_range = self._load_gpu_range()

    @contextmanager
    def section(self, name: str, *, enabled: bool) -> Iterator[None]:
        if not enabled:
            yield
            return

        with ExitStack() as stack:
            stack.enter_context(
                self.monitor.timer(f"profile.{name}", synchronize=True)
            )
            if self._gpu_range is not None:
                stack.enter_context(self._gpu_range(name))
            yield

    def start_run(self) -> None:
        if self.config.profile_python:
            self._profile = cProfile.Profile()
            self._profile.enable()
            self._python_profile_enabled = True

        if self.config.profile_memory:
            try:
                if tracemalloc.is_tracing():
                    self._tracemalloc_started_here = False
                else:
                    tracemalloc.start()
                    self._tracemalloc_started_here = True
                self._memory_profile_enabled = True
            except RuntimeError as exc:
                logger.warning("failed to start tracemalloc: %s", exc)

    def stop_run(self) -> None:
        if self._profile is not None:
            self._profile.disable()

        if self._memory_profile_enabled:
            try:
                current, peak = tracemalloc.get_traced_memory()
                self.monitor.set_metric("memory.python_traced.current_bytes", current)
                self.monitor.set_metric("memory.python_traced.peak_bytes", peak)
            except RuntimeError as exc:
                logger.warning("failed to collect tracemalloc metrics: %s", exc)
            finally:
                if self._tracemalloc_started_here:
                    tracemalloc.stop()
                self._memory_profile_enabled = False

    def dump_artifacts(self, output_dir: str | Path) -> list[Path]:
        if not self._python_profile_enabled or self._profile is None:
            return []

        output_path = Path(output_dir)
        artifacts = [
            output_path / "python_profile.prof",
            output_path / "python_profile.txt",
        ]

        try:
            output_path.mkdir(parents=True, exist_ok=True)
            self._profile.dump_stats(str(artifacts[0]))
            with artifacts[1].open("w", encoding="utf-8") as file:
                stats = pstats.Stats(self._profile, stream=file)
                stats.sort_stats("cumulative").print_stats(100)
        except OSError as exc:
            logger.warning("failed to dump profiling artifacts: %s", exc)
            return []

        return artifacts

    def _load_gpu_range(self):
        if (
            not self.config.profile_gpu_ranges
            or self.backend_profiler.name != "cupy"
        ):
            return None

        try:
            from cupyx.profiler import time_range
        except ImportError as exc:
            logger.warning("failed to enable NVTX ranges: %s", exc)
            return None

        return time_range
