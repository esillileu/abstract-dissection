from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentResult:
    """Portable result returned by an experiment before any sink persists it."""

    metrics: dict[str, float]
    artifact_root: Path
    model: Any | None = None
    artifacts: tuple[Path, ...] = field(default_factory=tuple)
    metric_rows: tuple[tuple[int, str, float], ...] = field(default_factory=tuple)
    profiling_metrics: dict[str, int | float] = field(default_factory=dict)
