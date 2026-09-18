from __future__ import annotations

from pathlib import Path

from dlfs.analysis.input import AnalysisRun
from dlfs.identity import Variant
from repro_core.results import ArtifactReference, MetricSeries, NativeRunResult


def _analysis_run_to_dict(study_id: str, run: AnalysisRun) -> dict[str, object]:
    return {
        "study_id": study_id,
        "run_id": run.run_id,
        "canonical_condition_id": run.canonical_condition_id,
        "native_condition_id": run.native_condition_id,
        "seed": run.seed,
        "variant": run.variant.value,
        "result": _native_result_to_dict(run.result),
        "local_artifact_root": (
            None if run.local_artifact_root is None else str(run.local_artifact_root)
        ),
    }


def _analysis_run_from_dict(item: dict[str, object]) -> AnalysisRun:
    local_root = item.get("local_artifact_root")
    return AnalysisRun(
        run_id=str(item["run_id"]),
        canonical_condition_id=str(item["canonical_condition_id"]),
        native_condition_id=str(item["native_condition_id"]),
        seed=str(item["seed"]),
        variant=Variant(str(item["variant"])),
        result=_native_result_from_dict(item["result"]),
        local_artifact_root=None if local_root is None else Path(str(local_root)),
    )


def _native_result_to_dict(result: NativeRunResult) -> dict[str, object]:
    return {
        "run_id": result.run_id,
        "schema_name": result.schema_name,
        "schema_version": result.schema_version,
        "protocol_version": result.protocol_version,
        "metrics": [
            {
                "metric_id": metric.metric_id,
                "unit": metric.unit,
                "split": metric.split,
                "axis": metric.axis,
                "steps": list(metric.steps),
                "values": list(metric.values),
            }
            for metric in result.metrics
        ],
        "artifacts": [vars(artifact) for artifact in result.artifacts],
        "artifact_aliases": dict(result.artifact_aliases),
        "provenance": dict(result.provenance),
        "provenance_ref": result.provenance_ref,
    }


def _native_result_from_dict(item: dict[str, object]) -> NativeRunResult:
    return NativeRunResult(
        run_id=str(item["run_id"]),
        schema_name=str(item["schema_name"]),
        schema_version=int(item["schema_version"]),
        protocol_version=str(item["protocol_version"]),
        metrics=tuple(
            MetricSeries(
                metric_id=str(metric["metric_id"]),
                unit=str(metric["unit"]),
                split=str(metric["split"]),
                axis=str(metric["axis"]),
                steps=tuple(metric["steps"]),
                values=tuple(float(value) for value in metric["values"]),
            )
            for metric in item.get("metrics", [])
        ),
        artifacts=tuple(
            ArtifactReference(**artifact) for artifact in item.get("artifacts", [])
        ),
        artifact_aliases=dict(item.get("artifact_aliases", {})),
        provenance=dict(item.get("provenance", {})),
        provenance_ref=item.get("provenance_ref"),
    )
