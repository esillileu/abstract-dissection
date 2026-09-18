"""CSV and observation artifact writing for DS2 training records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ._csv import append_csv

if TYPE_CHECKING:
    from ._records import DS2Records


def write_records_csv(records: DS2Records, artifact_root: Path) -> None:
    if records._written_root != artifact_root:
        records._written_rows.clear()
        records._written_root = artifact_root
    records.artifact_root = artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    records._materialize_pending_scalars()
    _append(
        records,
        "updates",
        artifact_root / "updates.csv",
        records.updates,
        columns=["update", "epoch", "batch_size", "loss", "book_loss", "lr"],
    )
    _append(
        records,
        "evaluations",
        artifact_root / "evaluations.csv",
        records.evaluations,
        columns=[
            "axis",
            "axis_step",
            "update",
            "epoch",
            "evaluation_set_id",
            "split",
            "unit",
            "unit_count",
            "metric",
            "value",
        ],
    )
    _append(
        records,
        "checkpoints",
        artifact_root / "checkpoints.csv",
        records.checkpoints,
        columns=[
            "update",
            "epoch",
            "kind",
            "path",
            "sha256",
            "checkpoint_id",
            "selection_metric",
            "selection_value",
        ],
    )
    _append(
        records,
        "timing_windows",
        artifact_root / "timing_windows.csv",
        [
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
            for item in records.timing_windows
        ],
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
    observations = artifact_root / "observations"
    observations.mkdir(exist_ok=True)
    _append(
        records,
        "source_samples",
        observations / "source_objectives.csv",
        records.source_samples,
        columns=[
            "update",
            "epoch",
            "local_iteration",
            "objective",
            "book_objective",
            "unit_count",
        ],
    )
    _append(
        records,
        "source_curves",
        observations / "source_curves.csv",
        records.source_curves,
        columns=[
            "series_id",
            "plot_index",
            "update_start",
            "update_end",
            "epoch_start",
            "epoch_end",
            "unit",
            "unit_count",
            "metric",
            "reducer",
            "value",
        ],
    )
    _append(
        records,
        "predictions",
        observations / "predictions.csv",
        records.predictions,
        columns=[
            "epoch",
            "example_id",
            "source",
            "target",
            "prediction",
            "exact_match",
            "token_correct",
            "token_count",
        ],
    )
    _append(
        records,
        "attention",
        observations / "attention.csv",
        records.attention,
        columns=["example_id", "decode_step", "encoder_position", "weight"],
    )
    if records.attention_render is not None:
        (observations / "attention_render.json").write_text(
            json.dumps(records.attention_render, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    records._pending_rows = 0


def _append(
    records: DS2Records,
    name: str,
    path: Path,
    rows: list[dict[str, object]],
    *,
    columns: list[str],
) -> None:
    append_csv(name, path, rows, columns=columns, written_rows=records._written_rows)
