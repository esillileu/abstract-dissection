"""Domain-neutral native result and storage contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True)
class MetricSeries:
    metric_id: str
    unit: str
    split: str
    axis: str
    steps: tuple[int | float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.steps) != len(self.values):
            raise ValueError("metric steps and values must have equal length")


@dataclass(frozen=True)
class ArtifactReference:
    path: str
    media_type: str | None = None
    sha256: str | None = None
    size: int | None = None
    available: bool = True
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.available and self.unavailable_reason is not None:
            raise ValueError("available artifact cannot have an unavailable reason")
        if not self.available and not self.unavailable_reason:
            raise ValueError("unavailable artifact requires a reason")


@dataclass(frozen=True)
class NativeRunResult:
    run_id: str
    schema_name: str
    schema_version: int
    protocol_version: str
    metrics: tuple[MetricSeries, ...]
    artifacts: tuple[ArtifactReference, ...] = ()
    artifact_aliases: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)
    provenance_ref: str | None = None

    def metric(self, metric_id: str) -> MetricSeries | None:
        return next((item for item in self.metrics if item.metric_id == metric_id), None)


class ResultStore(Protocol):
    def load(self, run_id: str) -> NativeRunResult: ...

    def list_run_ids(self, **selection: object) -> Sequence[str]: ...
