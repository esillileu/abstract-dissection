"""MLflow metric serialization for DS2 training records."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._csv import source_curve_metric_name

if TYPE_CHECKING:
    from ._records import DS2Records


def records_mlflow_metric_rows(
    records: DS2Records,
) -> tuple[tuple[int, str, float], ...]:
    records._materialize_pending_scalars()
    rows: list[tuple[int, str, float]] = []
    for row in records.updates:
        rows.append((int(row["update"]), "update/train/loss", float(row["loss"])))
        if row.get("book_loss") is not None:
            rows.append(
                (
                    int(row["update"]),
                    "update/train/book_loss",
                    float(row["book_loss"]),
                )
            )
        if isinstance(row["lr"], float):
            rows.append((int(row["update"]), "update/train/lr", row["lr"]))
    for row in records.evaluations:
        rows.append(
            (
                int(row["axis_step"]),
                f"{row['axis']}/eval_{row['split']}/{row['metric']}",
                float(row["value"]),
            )
        )
    for row in records.source_curves:
        metric_name = source_curve_metric_name(str(row.get("metric", "")))
        if metric_name is not None:
            rows.append((int(row["plot_index"]), metric_name, float(row["value"])))
    for window in records.timing_windows:
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
