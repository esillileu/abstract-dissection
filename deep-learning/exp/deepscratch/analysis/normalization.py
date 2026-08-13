"""Experiment-declared normalization into a cross-variant comparison view."""

from __future__ import annotations

from dataclasses import dataclass
from exp.framework.results import NativeRunResult

from ..identity import DeepScratchCoordinate
from .declarations import MetricDeclaration


@dataclass(frozen=True)
class ComparableMetric:
    metric_id: str
    unit: str
    split: str
    axis: str
    evaluation_protocol: str
    native_metric_id: str
    value_scale: float = 1.0


@dataclass(frozen=True)
class ComparisonObservation:
    coordinate: DeepScratchCoordinate
    metric_id: str
    unit: str
    split: str
    axis: str
    evaluation_protocol: str
    source_run_id: str
    native_schema: str
    provenance_ref: str | None
    steps: tuple[int | float, ...]
    values: tuple[float, ...]
    available: bool
    unavailable_reason: str | None = None


NormalizedObservation = ComparisonObservation


def normalize_declared_metric(
    coordinate: DeepScratchCoordinate,
    result: NativeRunResult,
    declaration: MetricDeclaration,
) -> ComparisonObservation:
    """Normalize one suite declaration without guessing absent values."""
    native_ids = declaration.native_ids(coordinate.variant)
    base = ComparableMetric(
        metric_id=declaration.metric_id,
        unit=declaration.unit,
        split=declaration.split,
        axis=declaration.axis,
        evaluation_protocol=result.protocol_version,
        native_metric_id=next(
            (item for item in native_ids if result.metric(item) is not None),
            native_ids[0],
        ),
        value_scale=declaration.value_scale,
    )
    if result.protocol_version not in declaration.protocols:
        return _unavailable(
            coordinate,
            result,
            base,
            "evaluation protocol is not declared comparable: "
            f"{result.protocol_version}",
        )
    return normalize_metric(coordinate, result, base)


def normalize_metric(
    coordinate: DeepScratchCoordinate,
    result: NativeRunResult,
    declaration: ComparableMetric,
) -> ComparisonObservation:
    if result.protocol_version != declaration.evaluation_protocol:
        return _unavailable(
            coordinate,
            result,
            declaration,
            "evaluation protocol mismatch: "
            f"{result.protocol_version} != {declaration.evaluation_protocol}",
        )
    native = next(
        (metric for metric in result.metrics if metric.metric_id == declaration.native_metric_id),
        None,
    )
    if native is None:
        return _unavailable(
            coordinate, result, declaration,
            f"native metric is absent: {declaration.native_metric_id}",
        )
    if (native.unit, native.split, native.axis) != (
        declaration.unit, declaration.split, declaration.axis
    ):
        return _unavailable(
            coordinate, result, declaration,
            "native metric unit/split/axis is not comparable",
        )
    return ComparisonObservation(
        coordinate=coordinate,
        metric_id=declaration.metric_id,
        unit=declaration.unit,
        split=declaration.split,
        axis=declaration.axis,
        evaluation_protocol=declaration.evaluation_protocol,
        source_run_id=result.run_id,
        native_schema=result.schema_name,
        provenance_ref=result.provenance_ref,
        steps=native.steps,
        values=tuple(value * declaration.value_scale for value in native.values),
        available=True,
    )


def _unavailable(
    coordinate: DeepScratchCoordinate,
    result: NativeRunResult,
    declaration: ComparableMetric,
    reason: str,
) -> ComparisonObservation:
    return ComparisonObservation(
        coordinate=coordinate,
        metric_id=declaration.metric_id,
        unit=declaration.unit,
        split=declaration.split,
        axis=declaration.axis,
        evaluation_protocol=declaration.evaluation_protocol,
        source_run_id=result.run_id,
        native_schema=result.schema_name,
        provenance_ref=result.provenance_ref,
        steps=(),
        values=(),
        available=False,
        unavailable_reason=reason,
    )
