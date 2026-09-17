from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from repro_core.context.events import TrainingWindowEvent


def project_mlflow_metric_rows(
    updates: list[dict[str, object]],
    evaluations: list[dict[str, object]],
    timing_windows: list[TrainingWindowEvent],
) -> tuple[tuple[int, str, float], ...]:
    """Project canonical CSV-owned records to MLflow metric rows."""
    rows: list[tuple[int, str, float]] = []
    for row in updates:
        rows.append((int(row["update"]), "update/train/loss", float(row["loss"])))
        lr = row["lr"]
        if isinstance(lr, float):
            rows.append((int(row["update"]), "update/train/lr", lr))
    for row in evaluations:
        for metric in ("loss", "accuracy"):
            value = row[metric]
            if value is not None:
                rows.append(
                    (
                        int(row["axis_step"]),
                        f"{row['axis']}/eval_{row['split']}/{metric}",
                        float(value),
                    )
                )
    for window in timing_windows:
        rows.append(
            (
                window.end_update,
                "update/runtime/window/train_wall_time_ms",
                window.train_wall_time_ns / 1_000_000,
            )
        )
        if window.train_device_time_ns is not None:
            rows.append(
                (
                    window.end_update,
                    "update/runtime/window/train_device_time_ms",
                    window.train_device_time_ns / 1_000_000,
                )
            )
        if window.eval_wall_time_ns is not None:
            rows.append(
                (
                    window.end_update,
                    "update/runtime/window/eval_wall_time_ms",
                    window.eval_wall_time_ns / 1_000_000,
                )
            )
        if window.eval_device_time_ns is not None:
            rows.append(
                (
                    window.end_update,
                    "update/runtime/window/eval_device_time_ms",
                    window.eval_device_time_ns / 1_000_000,
                )
            )
    return tuple(rows)
