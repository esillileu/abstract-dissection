from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from repro_core.context.events import (
    EpochEvent,
    TrainEndEvent,
    TrainingWindowEvent,
    UpdateEvent,
)

from .metrics import project_mlflow_metric_rows
from .serialization import (
    csv_value,
    materialize_pending_scalars,
    materialize_scalars,
    write_csv_file,
    write_csv_records,
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
        self.updates.append(
            {
                "update": event.update,
                "epoch": event.epoch,
                "batch_size": event.batch_size,
                "loss": event.loss,
                "lr": event.learning_rate,
            }
        )
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
        self.evaluations.append(
            {
                "axis": axis,
                "axis_step": axis_step,
                "update": update,
                "epoch": epoch,
                "evaluation_set_id": evaluation_set_id,
                "split": split,
                "example_count": result.example_count,
                "loss": result.loss,
                "accuracy": result.accuracy,
            }
        )
        self._mark_dirty()

    def add_timing_window(self, event: TrainingWindowEvent) -> None:
        self.timing_windows.append(event)
        self._mark_dirty()

    def add_checkpoint(
        self, *, update: int, epoch: int, kind: str, path: Path, sha256: str
    ) -> None:
        self.checkpoints.append(
            {
                "update": update,
                "epoch": epoch,
                "kind": kind,
                "path": str(path),
                "sha256": sha256,
            }
        )
        self._mark_dirty()

    def mlflow_metric_rows(self) -> tuple[tuple[int, str, float], ...]:
        """Project canonical CSV-owned records to MLflow metric rows."""
        self._materialize_pending_scalars()
        return project_mlflow_metric_rows(
            self.updates, self.evaluations, self.timing_windows
        )

    def write_csv(self, artifact_root: Path) -> None:
        """Durably materialize the schema-owned raw CSV artifacts."""
        self.artifact_root = artifact_root
        write_csv_records(
            artifact_root,
            updates=self.updates,
            evaluations=self.evaluations,
            checkpoints=self.checkpoints,
            timing_windows=self.timing_windows,
        )
        self._pending_rows = 0

    def flush(self) -> None:
        if self.artifact_root is not None:
            self.write_csv(self.artifact_root)

    def _mark_dirty(self) -> None:
        self._pending_rows += 1
        if self._pending_rows >= self.flush_interval:
            self.flush()

    def _materialize_pending_scalars(self) -> None:
        materialize_pending_scalars(self.updates)

    @staticmethod
    def _materialize_scalars(entries: list[tuple[dict[str, object], str]]) -> None:
        materialize_scalars(entries)

    @staticmethod
    def _write(
        path: Path, rows: list[dict[str, object]], *, columns: list[str] | None = None
    ) -> None:
        write_csv_file(path, rows, columns=columns)

    @staticmethod
    def _csv_value(value: object) -> object:
        return csv_value(value)
