"""DS2 RunSpec dataclasses and specification contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from repro_core.execution.spec import RunIdentity


@dataclass(frozen=True)
class SourceCurveSpec:
    kind: str
    every_updates: int | None = None
    every_epochs: int | None = None
    reducer: str = "mean"
    plot_index: str = "zero_based_append"


@dataclass(frozen=True)
class EvaluationTrigger:
    axis: Literal["epoch", "terminal"]
    sources: tuple[str, ...]
    every: int | None = None


@dataclass(frozen=True)
class RunSpec:
    kind: Literal[
        "word2vec",
        "language_modeling",
        "seq2seq",
        "observation",
        "performance_profile",
        "count_based_embedding",
    ]
    identity: RunIdentity
    atomic_run_id: str
    seed_policy: dict[str, object]
    dataset: dict[str, object]
    model: dict[str, object]
    optimizer: dict[str, object]
    loader: dict[str, object]
    budget: dict[str, object]
    recording: dict[str, object]
    source_curve: SourceCurveSpec | None
    evaluations: tuple[EvaluationTrigger, ...]
    checkpoint: dict[str, object]
    tracking: dict[str, object]
    numerics: dict[str, object]
    profiling: dict[str, object]
    scheduler: dict[str, object]
    objective: dict[str, object]
    path: Path
    protocol_version: str = "legacy"

    @property
    def config(self) -> dict[str, object]:
        return self.to_executor_config()

    def to_executor_config(self) -> dict[str, object]:
        from .executor_config import build_executor_config

        return build_executor_config(self)
