"""PF02 generic-axis vocabulary-size scaling study implementation."""

from __future__ import annotations

from dlfs.profile import (
    MeasurementProtocol,
    ProfilePoint,
    ProfileStudyResult,
    ScalingAxis,
)
from dlfs.profile.engine import measure_update_workload

from ..word2vec.adapters import (
    build_scaling_workload,
    default_vocabulary_sizes,
    make_scaling_backend,
    scaling_crossovers,
    scaling_environment,
)
from ..word2vec.contracts import Word2VecCondition


class Word2VecAxisScalingStudy:
    def run(self, config, context) -> ProfileStudyResult:
        del context
        profiling = _mapping(config, "profiling")
        protocol = MeasurementProtocol.from_mapping(profiling)
        axis = ScalingAxis.from_mapping(_mapping(profiling, "axis"))
        if axis.name != "vocabulary_size":
            raise ValueError("PF02 requires the vocabulary_size scaling axis")
        condition = Word2VecCondition.from_mapping(_mapping(profiling, "condition"))
        if condition.subject_variant != "implemented":
            raise ValueError("PF02 currently supports implemented workloads only")
        numerics = _mapping(config, "numerics")
        device = str(numerics.get("device", "cpu"))
        backend = make_scaling_backend(
            device=device,
            dtype=str(numerics.get("dtype", "float32")),
            seed=int(config.get("seed", 0)),
        )
        values = (
            tuple(int(value) for value in axis.values)
            if axis.values
            else default_vocabulary_sizes(device)
        )
        if not values or min(values) < 2 or len(values) != len(set(values)):
            raise ValueError("vocabulary sizes must be unique integers of at least 2")
        if axis.reverse:
            values = tuple(reversed(values))
        batch_size = int(_mapping(config, "loader").get("batch_size", 100))
        points = tuple(
            _measure_point(
                config,
                condition=condition,
                vocabulary_size=value,
                backend=backend,
                batch_size=batch_size,
                protocol=protocol,
            )
            for value in values
        )
        raw_rows = [_crossover_row(condition, point) for point in points]
        ok_count = sum(point.status == "ok" for point in points)
        return ProfileStudyResult(
            study_id="e11",
            group_id="PF02",
            study_kind="axis_scaling",
            source_study="e02",
            protocol_version=str(config.get("protocol_version", "legacy")),
            schema_name="ds2-profile",
            points=points,
            environment=scaling_environment(backend),
            metadata={
                "condition": condition.legacy_id(),
                "axis": axis.__dict__,
                "measurement_protocol": protocol.__dict__,
                "measured_update_count": protocol.measured_iterations * ok_count,
                "samples_seen": protocol.measured_iterations * ok_count * batch_size,
            },
            derived={"crossovers": scaling_crossovers(raw_rows)},
        )


def _measure_point(
    config,
    *,
    condition,
    vocabulary_size,
    backend,
    batch_size,
    protocol,
) -> ProfilePoint:
    try:
        workload = build_scaling_workload(
            condition,
            vocabulary_size=vocabulary_size,
            backend=backend,
            batch_size=batch_size,
            update_count=(
                1
                + protocol.warmup_iterations
                + protocol.measured_iterations
                + protocol.measured_iterations * protocol.repetitions
            ),
        )
    except Exception as error:
        if not _is_out_of_memory(error):
            raise
        return ProfilePoint(
            condition_id=str(config["atomic_run_id"]),
            axes={"vocabulary_size": vocabulary_size},
            status="out_of_memory",
            metrics={},
            error=f"{type(error).__name__}: {error}",
        )
    measured = measure_update_workload(
        str(config["atomic_run_id"]),
        workload,
        axes={"vocabulary_size": vocabulary_size},
        protocol=protocol,
    )
    return ProfilePoint(
        condition_id=measured.condition_id,
        axes=measured.axes,
        status=measured.status,
        metrics={
            **measured.metrics,
            "dense_parameter_optimizer_bytes": (8 * vocabulary_size * 100 * 4),
        },
        timings=measured.timings,
        error=measured.error,
    )


def _crossover_row(condition, point) -> dict[str, object]:
    objective = {
        "full_softmax": "FullSoftmax",
        "negative_sampling": "NegativeSampling",
        "fused_negative_sampling": "FusedNegativeSampling",
    }[condition.objective]
    return {
        "model": "CBOW" if condition.model == "cbow" else "SkipGram",
        "objective": objective,
        "status": point.status,
        "vocab_size": point.axes["vocabulary_size"],
        "update_ms": point.metrics.get("update_ms"),
        "ci95_lower_ms": point.metrics.get("ci95_lower_ms"),
        "ci95_upper_ms": point.metrics.get("ci95_upper_ms"),
    }


def _is_out_of_memory(error: Exception) -> bool:
    return isinstance(error, MemoryError) or type(error).__name__ == "OutOfMemoryError"


def _mapping(values, key):
    child = values.get(key, {})
    if not isinstance(child, dict):
        raise ValueError(f"{key} must be a mapping")
    return child


STUDY = Word2VecAxisScalingStudy()
