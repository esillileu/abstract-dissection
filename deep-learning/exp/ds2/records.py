"""DS2's language/sequence projection of trainer events to raw records."""

from __future__ import annotations

import csv
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
    timing_windows: list[TrainingWindowEvent] = field(default_factory=list)
    epochs: list[EpochEvent] = field(default_factory=list)
    end: TrainEndEvent | None = None

    def on_update(self, event: UpdateEvent) -> None:
        self.updates.append({"update": event.update, "epoch": event.epoch, "batch_size": event.batch_size, "loss": event.loss, "lr": event.learning_rate})

    def on_source_objective(self, event: SourceObjectiveSample) -> None:
        self.source_samples.append({"update": event.update, "epoch": event.epoch, "local_iteration": event.local_iteration, "objective": event.objective, "unit_count": event.unit_count})

    def add_source_curve(self, point: dict[str, object]) -> None:
        self.source_curves.append(point)

    def on_epoch(self, event: EpochEvent) -> None:
        self.epochs.append(event)

    def on_train_end(self, event: TrainEndEvent) -> None:
        self.end = event

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

    def add_timing_window(self, event: TrainingWindowEvent) -> None:
        self.timing_windows.append(event)

    def history_rows(self) -> tuple[tuple[str, int, str, float], ...]:
        rows: list[tuple[str, int, str, float]] = []
        for row in self.updates:
            loss = row["loss"]
            rows.append(("update", int(row["update"]), "train/loss", loss.backend.scalar_to_float(loss.data)))
            if isinstance(row["lr"], float):
                rows.append(("update", int(row["update"]), "train/lr", row["lr"]))
        for row in self.evaluations:
            rows.append((str(row["axis"]), int(row["axis_step"]), f"eval_{row['split']}/{row['metric']}", float(row["value"])))
        for window in self.timing_windows:
            rows.append(("update", window.end_update, "runtime/window/train_wall_time_ms", window.train_wall_time_ns / 1_000_000))
            if window.eval_wall_time_ns is not None:
                rows.append(("update", window.end_update, "runtime/window/eval_wall_time_ms", window.eval_wall_time_ns / 1_000_000))
        return tuple(rows)

    def write_csv(self, artifact_root: Path) -> None:
        artifact_root.mkdir(parents=True, exist_ok=True)
        self._write(artifact_root / "updates.csv", self.updates)
        self._write(artifact_root / "evaluations.csv", self.evaluations)
        self._write(artifact_root / "timing_windows.csv", [{"start_update": item.start_update, "end_update": item.end_update, "update_count": item.update_count, "closed_by": item.closed_by, "train_wall_time_ns": item.train_wall_time_ns, "train_device_time_ns": item.train_device_time_ns, "eval_wall_time_ns": item.eval_wall_time_ns, "eval_device_time_ns": item.eval_device_time_ns} for item in self.timing_windows])
        observations = artifact_root / "observations"
        observations.mkdir(exist_ok=True)
        self._write(observations / "source_curves.csv", self.source_curves)

    @staticmethod
    def _write(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: DS2Records._csv_value(value) for key, value in row.items()})

    @staticmethod
    def _csv_value(value: object) -> object:
        if hasattr(value, "backend") and hasattr(value, "data"):
            return value.backend.scalar_to_float(value.data)
        return value
