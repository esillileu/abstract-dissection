"""Projection of canonical profile results to artifacts and MLflow scalars."""

from __future__ import annotations

import json
from pathlib import Path

from repro_core.context.contracts import ExperimentResult

from .contracts import ProfileStudyResult


def to_experiment_result(
    result: ProfileStudyResult,
    *,
    artifact_root: Path,
) -> ExperimentResult:
    output = artifact_root / "profile" / "result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ok_points = [point for point in result.points if point.status == "ok"]
    metrics: dict[str, float] = {
        "final/status/success": 1.0,
        "final/status/nan_detected": 0.0,
        "final/status/inf_detected": 0.0,
        "final/status/diverged": 0.0,
        "final/system/total_updates": float(
            result.metadata.get("measured_update_count", 0)
        ),
        "final/system/completed_epochs": 0.0,
        "final/system/samples_seen": float(result.metadata.get("samples_seen", 0)),
        "profile/points/ok": float(len(ok_points)),
        "profile/points/out_of_memory": float(
            sum(point.status == "out_of_memory" for point in result.points)
        ),
    }
    for point in ok_points:
        axis_suffix = "/".join(
            f"{name}/{value}" for name, value in sorted(point.axes.items())
        )
        prefix = "profile" if not axis_suffix else f"profile/{axis_suffix}"
        for name, value in point.metrics.items():
            if value is not None:
                metrics[f"{prefix}/{name}"] = float(value)
    return ExperimentResult(
        metrics=metrics,
        artifact_root=artifact_root,
        profiling_metrics={
            "profile.points.ok": len(ok_points),
            "profile.points.out_of_memory": sum(
                point.status == "out_of_memory" for point in result.points
            ),
        },
    )
