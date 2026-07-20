from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from .contracts import ExperimentResult


@dataclass
class ExperimentContext:
    """Sink-neutral runtime services made available to every executor."""

    emit_metric: Callable[[int, dict[str, float]], None] = lambda step, metrics: None
    metadata: dict[str, object] = field(default_factory=dict)


class ExperimentExecutor(Protocol):
    def run(self, config: dict[str, object], context: ExperimentContext) -> ExperimentResult: ...
