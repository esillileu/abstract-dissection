from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from repro_core.context.events import TrainingWindowEvent


def materialize_pending_scalars(updates: list[dict[str, object]]) -> None:
    materialize_scalars([(row, "loss") for row in updates])


def materialize_scalars(entries: list[tuple[dict[str, object], str]]) -> None:
    groups: dict[int, tuple[object, list[tuple[dict[str, object], str, object]]]] = {}
    for row, key in entries:
        value = row.get(key)
        if not hasattr(value, "backend") or not hasattr(value, "data"):
            continue
        group = groups.setdefault(id(value.backend), (value.backend, []))
        group[1].append((row, key, value))
    for backend, values in groups.values():
        stacked = backend.xp.stack([value.data.reshape(()) for _, _, value in values])
        host_values = backend.to_numpy(stacked)
        for (row, key, _), host_value in zip(values, host_values, strict=True):
            row[key] = float(host_value)


def write_csv_file(
    path: Path, rows: list[dict[str, object]], *, columns: list[str] | None = None
) -> None:
    columns = columns or (list(rows[0]) if rows else [])
    if not columns:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def csv_value(value: object) -> object:
    if hasattr(value, "backend") and hasattr(value, "data"):
        return value.backend.scalar_to_float(value.data)
    return value


def write_csv_records(
    artifact_root: Path,
    *,
    updates: list[dict[str, object]],
    evaluations: list[dict[str, object]],
    checkpoints: list[dict[str, object]],
    timing_windows: list[TrainingWindowEvent],
) -> None:
    """Durably materialize the schema-owned raw CSV artifacts."""
    artifact_root.mkdir(parents=True, exist_ok=True)
    materialize_pending_scalars(updates)
    write_csv_file(
        artifact_root / "updates.csv",
        updates,
        columns=["update", "epoch", "batch_size", "loss", "lr"],
    )
    write_csv_file(
        artifact_root / "evaluations.csv",
        evaluations,
        columns=[
            "axis",
            "axis_step",
            "update",
            "epoch",
            "evaluation_set_id",
            "split",
            "example_count",
            "loss",
            "accuracy",
        ],
    )
    write_csv_file(
        artifact_root / "checkpoints.csv",
        checkpoints,
        columns=["update", "epoch", "kind", "path", "sha256"],
    )
    windows = [
        {
            "start_update": item.start_update,
            "end_update": item.end_update,
            "update_count": item.update_count,
            "closed_by": item.closed_by,
            "train_wall_time_ns": item.train_wall_time_ns,
            "train_device_time_ns": item.train_device_time_ns,
            "eval_wall_time_ns": item.eval_wall_time_ns,
            "eval_device_time_ns": item.eval_device_time_ns,
        }
        for item in timing_windows
    ]
    write_csv_file(
        artifact_root / "timing_windows.csv",
        windows,
        columns=[
            "start_update",
            "end_update",
            "update_count",
            "closed_by",
            "train_wall_time_ns",
            "train_device_time_ns",
            "eval_wall_time_ns",
            "eval_device_time_ns",
        ],
    )
