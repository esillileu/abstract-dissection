"""PF01 update-breakdown study implementation."""

from __future__ import annotations

from dlfs.profile import MeasurementProtocol, ProfilePoint, ProfileStudyResult
from dlfs.profile.engine import (
    measure_update_workload,
    measure_workload_sections,
)

from ..word2vec.adapters import build_module_workload, build_update_workload
from ..word2vec.contracts import Word2VecCondition


class Word2VecUpdateBreakdownStudy:
    def run(self, config, context) -> ProfileStudyResult:
        del context
        profiling = _mapping(config, "profiling")
        condition = Word2VecCondition.from_mapping(_mapping(profiling, "condition"))
        protocol = MeasurementProtocol.from_mapping(profiling)
        loader = _mapping(config, "loader")
        training = _mapping(config, "training")
        numerics = _mapping(config, "numerics")
        batch_size = int(loader.get("batch_size", 100))
        workload, source, environment = build_update_workload(
            condition,
            device=str(numerics.get("device", "cpu")),
            batch_size=batch_size,
        )
        measured = measure_update_workload(
            str(config["atomic_run_id"]),
            workload,
            axes={},
            protocol=protocol,
        )
        modules = []
        if bool(profiling.get("measure_modules", True)):
            module_workload = build_module_workload(
                condition, source=source, batch_size=batch_size
            )
            scope = module_workload.measurement_scope
            section_results = measure_workload_sections(
                module_workload,
                warmup_iterations=int(profiling.get("module_warmup", 5)),
                measured_iterations=int(profiling.get("module_iterations", 20)),
            )
            modules = [
                {
                    "condition": condition.legacy_id(),
                    "component": component,
                    "measurement_scope": scope,
                    **result,
                }
                for component, result in section_results.items()
            ]
        mean_ms = float(measured.metrics["update_ms"])
        dataset_samples = int(source["dataset_samples"])
        epochs = int(training.get("max_epochs", 10))
        updates_per_epoch = dataset_samples // batch_size
        epoch_seconds = mean_ms * updates_per_epoch / 1_000
        point = ProfilePoint(
            condition_id=str(config["atomic_run_id"]),
            axes={},
            status=measured.status,
            metrics={
                "update/mean_ms": mean_ms,
                "update/stdev_ms": measured.metrics["update_stdev_ms"],
                "update/p50_ms": measured.metrics["p50_ms"],
                "update/p95_ms": measured.metrics["p95_ms"],
                "update/cold_ms": measured.metrics["cold_ms"],
                "update/window_mean_ms": measured.metrics["window_update_ms"],
                "update/event_mean_ms": measured.metrics["event_update_ms"],
                "epoch/estimated_seconds": epoch_seconds,
                "total/estimated_seconds": epoch_seconds * epochs,
            },
            timings=measured.timings,
            sections={"modules": modules},
            error=measured.error,
        )
        return ProfileStudyResult(
            study_id="e10",
            group_id="PF01",
            study_kind="update_breakdown",
            source_study="e02",
            protocol_version=str(config.get("protocol_version", "legacy")),
            schema_name="ds2-profile",
            points=(point,),
            environment=environment,
            metadata={
                "condition": condition.legacy_id(),
                "measurement_protocol": protocol.__dict__,
                "measured_update_count": protocol.measured_iterations,
                "samples_seen": protocol.measured_iterations * batch_size,
            },
        )


def _mapping(values, key):
    child = values.get(key, {})
    if not isinstance(child, dict):
        raise ValueError(f"{key} must be a mapping")
    return child


STUDY = Word2VecUpdateBreakdownStudy()
