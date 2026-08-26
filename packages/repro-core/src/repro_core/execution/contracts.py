"""Execution result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentResult:
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_root: Path | None = None
    model: Any = None
    artifacts: tuple[Path, ...] = ()
    metric_rows: tuple[Any, ...] = ()
    profiling_metrics: dict[str, Any] = field(default_factory=dict)


__all__ = ["ExperimentResult"]
