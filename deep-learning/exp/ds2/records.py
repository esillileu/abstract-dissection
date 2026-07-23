"""DS2's language/sequence projection of trainer events to raw records."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from mlprosection.events import EpochEvent, SourceObjectiveSample, TrainEndEvent, TrainingWindowEvent, UpdateEvent


@dataclass
class DS2Records:
    """DS2-owned long-form evaluation and observation artifacts."""

    updates: list[dict[str, object]] = field(default_factory=list)
    evaluations: list[dict[str, object]] = field(default_factory=list)
    source_samples: list[dict[str, object]] = field(default_factory=list)
    source_curves: list[dict[str, object]] = field(default_factory=list)
    checkpoints: list[dict[str, object]] = field(default_factory=list)
    predictions: list[dict[str, object]] = field(default_factory=list)
    attention: list[dict[str, object]] = field(default_factory=list)
    attention_render: dict[str, object] | None = None
    timing_windows: list[TrainingWindowEvent] = field(default_factory=list)
    epochs: list[EpochEvent] = field(default_factory=list)
    end: TrainEndEvent | None = None
    artifact_root: Path | None = None
    flush_interval: int = 256
    _pending_rows: int = 0
    _written_rows: dict[str, int] = field(default_factory=dict)
    _written_root: Path | None = None
    _materialized_rows: dict[str, int] = field(default_factory=dict)

    def bind_artifact_root(self, artifact_root: Path) -> None:
        if self._written_root != artifact_root:
            self._written_rows.clear()
            self._written_root = artifact_root
        self.artifact_root = artifact_root

    def on_update(self, event: UpdateEvent) -> None:
        self.updates.append({"update": event.update, "epoch": event.epoch, "batch_size": event.batch_size, "loss": event.loss, "lr": event.learning_rate})
        self._mark_dirty()

    def on_source_objective(self, event: SourceObjectiveSample) -> None:
        self.source_samples.append({"update": event.update, "epoch": event.epoch, "local_iteration": event.local_iteration, "objective": event.objective, "unit_count": event.unit_count})
        self._mark_dirty()

    def add_source_curve(self, point: dict[str, object]) -> None:
        self.source_curves.append(point)
        self._mark_dirty()

    def on_epoch(self, event: EpochEvent) -> None:
        self.epochs.append(event)
        self.flush()

    def on_train_end(self, event: TrainEndEvent) -> None:
        self.end = event
        self.flush()

    def add_evaluation(self, *, axis: str, axis_step: int, update: int, epoch: int, evaluation_set_id: str, split: str, result: object) -> None:
        metrics = {
            "loss": result.loss,
            "accuracy": result.accuracy,
            "perplexity": result.perplexity,
            "exact_match_accuracy": result.exact_match_accuracy,
            "token_accuracy": result.token_accuracy,
        }
        for metric, value in metrics.items():
            if value is not None:
                self.evaluations.append({"axis": axis, "axis_step": axis_step, "update": update, "epoch": epoch, "evaluation_set_id": evaluation_set_id, "split": split, "unit": result.unit, "unit_count": result.unit_count or result.example_count, "metric": metric, "value": value})
                self._mark_dirty()

    def add_timing_window(self, event: TrainingWindowEvent) -> None:
        self.timing_windows.append(event)
        self._mark_dirty()

    def add_checkpoint(
        self,
        *,
        update: int,
        epoch: int,
        kind: str,
        path: Path,
        sha256: str,
        checkpoint_id: str = "",
        selection_metric: str = "",
        selection_value: float | str = "",
    ) -> None:
        self.checkpoints.append({
            "update": update, "epoch": epoch, "kind": kind, "path": str(path),
            "sha256": sha256, "checkpoint_id": checkpoint_id,
            "selection_metric": selection_metric, "selection_value": selection_value,
        })
        self._mark_dirty()

    def add_prediction(self, row: dict[str, object]) -> None:
        self.predictions.append(row)
        self._mark_dirty()

    def add_attention(self, row: dict[str, object]) -> None:
        self.attention.append(row)
        self._mark_dirty()

    def set_attention_render(self, value: dict[str, object]) -> None:
        self.attention_render = value
        self._mark_dirty()

    def mlflow_metric_rows(self) -> tuple[tuple[int, str, float], ...]:
        self._materialize_pending_scalars()
        rows: list[tuple[int, str, float]] = []
        for row in self.updates:
            rows.append((int(row["update"]), "update/train/loss", float(row["loss"])))
            if isinstance(row["lr"], float):
                rows.append((int(row["update"]), "update/train/lr", row["lr"]))
        for row in self.evaluations:
            rows.append((int(row["axis_step"]), f"{row['axis']}/eval_{row['split']}/{row['metric']}", float(row["value"])))
        for row in self.source_curves:
            metric_name = _source_curve_metric_name(str(row.get("metric", "")))
            if metric_name is not None:
                rows.append((int(row["plot_index"]), metric_name, float(row["value"])))
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
        if self._written_root != artifact_root:
            self._written_rows.clear()
            self._written_root = artifact_root
        self.artifact_root = artifact_root
        artifact_root.mkdir(parents=True, exist_ok=True)
        self._materialize_pending_scalars()
        self._append("updates", artifact_root / "updates.csv", self.updates, columns=["update", "epoch", "batch_size", "loss", "lr"])
        self._append("evaluations", artifact_root / "evaluations.csv", self.evaluations, columns=["axis", "axis_step", "update", "epoch", "evaluation_set_id", "split", "unit", "unit_count", "metric", "value"])
        self._append("checkpoints", artifact_root / "checkpoints.csv", self.checkpoints, columns=["update", "epoch", "kind", "path", "sha256", "checkpoint_id", "selection_metric", "selection_value"])
        self._append("timing_windows", artifact_root / "timing_windows.csv", [{"start_update": item.start_update, "end_update": item.end_update, "update_count": item.update_count, "closed_by": item.closed_by, "train_wall_time_ns": item.train_wall_time_ns, "train_device_time_ns": item.train_device_time_ns, "eval_wall_time_ns": item.eval_wall_time_ns, "eval_device_time_ns": item.eval_device_time_ns} for item in self.timing_windows], columns=["start_update", "end_update", "update_count", "closed_by", "train_wall_time_ns", "train_device_time_ns", "eval_wall_time_ns", "eval_device_time_ns"])
        observations = artifact_root / "observations"
        observations.mkdir(exist_ok=True)
        self._append("source_samples", observations / "source_objectives.csv", self.source_samples, columns=["update", "epoch", "local_iteration", "objective", "unit_count"])
        self._append("source_curves", observations / "source_curves.csv", self.source_curves, columns=["series_id", "plot_index", "update_start", "update_end", "epoch_start", "epoch_end", "unit", "unit_count", "metric", "reducer", "value"])
        self._append("predictions", observations / "predictions.csv", self.predictions, columns=["epoch", "example_id", "source", "target", "prediction", "exact_match", "token_correct", "token_count"])
        self._append("attention", observations / "attention.csv", self.attention, columns=["example_id", "decode_step", "encoder_position", "weight"])
        if self.attention_render is not None:
            (observations / "attention_render.json").write_text(json.dumps(self.attention_render, indent=2, sort_keys=True), encoding="utf-8")
        self._pending_rows = 0

    def flush(self) -> None:
        if self.artifact_root is not None:
            self.write_csv(self.artifact_root)

    def _mark_dirty(self) -> None:
        self._pending_rows += 1
        if self._pending_rows >= self.flush_interval:
            self.flush()

    def _materialize_pending_scalars(self) -> None:
        collections = (
            ("updates", self.updates, "loss"),
            ("source_samples", self.source_samples, "objective"),
            ("source_curves", self.source_curves, "value"),
        )
        entries = []
        for name, rows, key in collections:
            start = self._materialized_rows.get(name, 0)
            entries.extend((row, key) for row in rows[start:])
            self._materialized_rows[name] = len(rows)
        self._materialize_scalars(entries)

    def _append(
        self,
        name: str,
        path: Path,
        rows: list[dict[str, object]],
        *,
        columns: list[str],
    ) -> None:
        initialized = name in self._written_rows
        start = self._written_rows.get(name, 0)
        if not initialized or not path.exists():
            start = 0
        pending = rows[start:]
        if initialized and not pending and path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if initialized and path.exists() else "w"
        with path.open(mode, newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            if mode == "w":
                writer.writeheader()
            for row in pending:
                writer.writerow({key: self._csv_value(row.get(key, "")) for key in columns})
        self._written_rows[name] = len(rows)

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
    def _csv_value(value: object) -> object:
        if hasattr(value, "backend") and hasattr(value, "data"):
            return value.backend.scalar_to_float(value.data)
        return value


def _source_curve_metric_name(metric: str) -> str | None:
    if metric == "loss":
        return "series/train/loss"
    if metric == "perplexity":
        return "series/train/perplexity"
    if metric == "exact_match_accuracy":
        return "series/eval_test/exact_match_accuracy"
    return None
