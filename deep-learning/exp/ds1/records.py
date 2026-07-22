"""DS1's MNIST-oriented projection of trainer events to raw records."""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
from pathlib import Path
from typing import Literal

from mlprosection.events import (
    EpochEvent,
    TrainEndEvent,
    TrainingWindowEvent,
    UpdateEvent,
)


@dataclass
class DS1Records:
    """DS1's CSV and MLflow metric representation.

    This is intentionally not part of ``src``: its loss/accuracy columns and
    metric names are properties of the DS1 experiment schema.
    """

    updates: list[dict[str, object]] = field(default_factory=list)
    evaluations: list[dict[str, object]] = field(default_factory=list)
    checkpoints: list[dict[str, object]] = field(default_factory=list)
    timing_windows: list[TrainingWindowEvent] = field(default_factory=list)
    epochs: list[EpochEvent] = field(default_factory=list)
    end: TrainEndEvent | None = None
    artifact_root: Path | None = None
    flush_interval: int = 256
    _pending_rows: int = 0

    def bind_artifact_root(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root

    def on_update(self, event: UpdateEvent) -> None:
        self.updates.append({
            "update": event.update,
            "epoch": event.epoch,
            "batch_size": event.batch_size,
            "loss": event.loss,
            "lr": event.learning_rate,
        })
        self._mark_dirty()

    def on_epoch(self, event: EpochEvent) -> None:
        self.epochs.append(event)
        self.flush()

    def on_train_end(self, event: TrainEndEvent) -> None:
        self.end = event
        self.flush()

    def add_evaluation(
        self,
        *,
        axis: Literal["update", "epoch", "terminal"],
        axis_step: int,
        update: int,
        epoch: int,
        evaluation_set_id: str,
        split: str,
        result,
    ) -> None:
        self.evaluations.append({
            "axis": axis, "axis_step": axis_step, "update": update,
            "epoch": epoch, "evaluation_set_id": evaluation_set_id,
            "split": split, "example_count": result.example_count,
            "loss": result.loss, "accuracy": result.accuracy,
        })
        self._mark_dirty()

    def add_timing_window(self, event: TrainingWindowEvent) -> None:
        self.timing_windows.append(event)
        self._mark_dirty()

    def add_checkpoint(self, *, update: int, epoch: int, kind: str, path: Path, sha256: str) -> None:
        self.checkpoints.append({
            "update": update, "epoch": epoch, "kind": kind,
            "path": str(path), "sha256": sha256,
        })
        self._mark_dirty()

    def mlflow_metric_rows(self) -> tuple[tuple[int, str, float], ...]:
        """Project canonical CSV-owned records to MLflow metric rows."""
        self._materialize_pending_scalars()
        rows: list[tuple[int, str, float]] = []
        for row in self.updates:
            rows.append((int(row["update"]), "update/train/loss", float(row["loss"])))
            lr = row["lr"]
            if isinstance(lr, float):
                rows.append((int(row["update"]), "update/train/lr", lr))
        for row in self.evaluations:
            for metric in ("loss", "accuracy"):
                value = row[metric]
                if value is not None:
                    rows.append((int(row["axis_step"]), f"{row['axis']}/eval_{row['split']}/{metric}", float(value)))
        for window in self.timing_windows:
            rows.append((window.end_update, "update/runtime/window/train_wall_time_ms", window.train_wall_time_ns / 1_000_000))
            if window.train_device_time_ns is not None:
                rows.append((window.end_update, "update/runtime/window/train_device_time_ms", window.train_device_time_ns / 1_000_000))
            if window.eval_wall_time_ns is not None:
                rows.append((window.end_update, "update/runtime/window/eval_wall_time_ms", window.eval_wall_time_ns / 1_000_000))
            if window.eval_device_time_ns is not None:
                rows.append((window.end_update, "update/runtime/window/eval_device_time_ms", window.eval_device_time_ns / 1_000_000))
        return tuple(rows)

    def write_csv(self, artifact_root: Path) -> None:
        """Durably materialize the schema-owned raw CSV artifacts."""
        self.artifact_root = artifact_root
        artifact_root.mkdir(parents=True, exist_ok=True)
        self._materialize_pending_scalars()
        self._write(artifact_root / "updates.csv", self.updates, columns=["update", "epoch", "batch_size", "loss", "lr"])
        self._write(artifact_root / "evaluations.csv", self.evaluations, columns=["axis", "axis_step", "update", "epoch", "evaluation_set_id", "split", "example_count", "loss", "accuracy"])
        self._write(artifact_root / "checkpoints.csv", self.checkpoints, columns=["update", "epoch", "kind", "path", "sha256"])
        windows = [
            {
                "start_update": item.start_update, "end_update": item.end_update,
                "update_count": item.update_count, "closed_by": item.closed_by,
                "train_wall_time_ns": item.train_wall_time_ns,
                "train_device_time_ns": item.train_device_time_ns,
                "eval_wall_time_ns": item.eval_wall_time_ns,
                "eval_device_time_ns": item.eval_device_time_ns,
            }
            for item in self.timing_windows
        ]
        self._write(artifact_root / "timing_windows.csv", windows, columns=["start_update", "end_update", "update_count", "closed_by", "train_wall_time_ns", "train_device_time_ns", "eval_wall_time_ns", "eval_device_time_ns"])
        self._pending_rows = 0

    def flush(self) -> None:
        if self.artifact_root is not None:
            self.write_csv(self.artifact_root)

    def _mark_dirty(self) -> None:
        self._pending_rows += 1
        if self._pending_rows >= self.flush_interval:
            self.flush()

    def _materialize_pending_scalars(self) -> None:
        self._materialize_scalars([(row, "loss") for row in self.updates])

    @staticmethod
    def _materialize_scalars(entries: list[tuple[dict[str, object], str]]) -> None:
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

    @staticmethod
    def _write(path: Path, rows: list[dict[str, object]], *, columns: list[str] | None = None) -> None:
        columns = columns or (list(rows[0]) if rows else [])
        if not columns:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: DS1Records._csv_value(value) for key, value in row.items()})

    @staticmethod
    def _csv_value(value: object) -> object:
        if hasattr(value, "backend") and hasattr(value, "data"):
            return value.backend.scalar_to_float(value.data)
        return value
