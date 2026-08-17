"""Portable contracts shared by profile executors and study implementations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections.abc import Callable
from typing import Literal, Mapping, Protocol

from mlprosection.experiment.executor import ExperimentContext


@dataclass(frozen=True)
class MeasurementProtocol:
    warmup_iterations: int
    measured_iterations: int
    repetitions: int
    timing_source: Literal["window", "event"] = "window"

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> MeasurementProtocol:
        protocol = cls(
            warmup_iterations=int(values.get("warmup_updates", 20)),
            measured_iterations=int(values.get("measured_updates", 50)),
            repetitions=int(values.get("repetitions", 5)),
            timing_source=str(values.get("timing_source", "window")),  # type: ignore[arg-type]
        )
        if protocol.warmup_iterations < 0:
            raise ValueError("profile warmup iterations must be non-negative")
        if min(protocol.measured_iterations, protocol.repetitions) < 1:
            raise ValueError("profile measurement and repetitions must be positive")
        if protocol.timing_source not in {"window", "event"}:
            raise ValueError("profile timing source must be window or event")
        return protocol


@dataclass(frozen=True)
class ScalingAxis:
    name: str
    values: tuple[int | float | str, ...] = ()
    schedule: str | None = None
    reverse: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> ScalingAxis:
        raw_values = values.get("values", ())
        if raw_values == "device_default":
            raw_values = ()
        if not isinstance(raw_values, list | tuple):
            raise ValueError("profile scaling axis values must be a list")
        axis = cls(
            name=str(values.get("name", "")),
            values=tuple(raw_values),
            schedule=None if values.get("schedule") is None else str(values["schedule"]),
            reverse=bool(values.get("reverse", False)),
        )
        if not axis.name:
            raise ValueError("profile scaling axis requires a name")
        return axis


@dataclass(frozen=True)
class ProfilePoint:
    condition_id: str
    axes: dict[str, int | float | str]
    status: Literal["ok", "out_of_memory", "failed"]
    metrics: dict[str, int | float | None]
    timings: dict[str, object] = field(default_factory=dict)
    sections: dict[str, object] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class ProfileSection:
    operation: Callable[[], object]
    prepare: Callable[[], object] | None = None


@dataclass(frozen=True)
class ProfileStudyResult:
    study_id: str
    group_id: str
    study_kind: str
    source_study: str | None
    protocol_version: str
    schema_name: str
    points: tuple[ProfilePoint, ...]
    environment: dict[str, object]
    metadata: dict[str, object] = field(default_factory=dict)
    derived: dict[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_name": self.schema_name,
            "schema_version": 1,
            **asdict(self),
        }


class ProfileStudy(Protocol):
    def run(
        self,
        config: dict[str, object],
        context: ExperimentContext,
    ) -> ProfileStudyResult: ...


class ProfileWorkload(Protocol):
    @property
    def backend(self): ...

    def update(self) -> None: ...

    def sections(self) -> Mapping[str, ProfileSection]: ...

    def metadata(self) -> Mapping[str, object]: ...

    def release(self) -> None: ...
