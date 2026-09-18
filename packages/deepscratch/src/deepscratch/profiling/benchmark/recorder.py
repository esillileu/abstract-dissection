"""Context-manager based execution section recorder."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

from .results import TimingStats


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


__all__ = ["SectionRecorder"]
