"""Dependency-neutral events emitted by deepscratch trainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from deepscratch.core import Tensor


@dataclass(frozen=True)
class UpdateEvent:
    """One completed optimizer update."""

    update: int
    epoch: int
    batch_size: int
    loss: Tensor
    learning_rate: float | tuple[float, ...]
    book_loss: Tensor | None = None


@dataclass(frozen=True)
class SourceObjectiveSample:
    """Pre-update objective retained to reconstruct a source curve."""

    update: int
    epoch: int
    local_iteration: int
    objective: Tensor
    unit_count: int
    book_objective: Tensor | None = None


@dataclass(frozen=True)
class EpochEvent:
    epoch: int
    start_update: int
    end_update: int
    sample_count: int


@dataclass(frozen=True)
class TrainStartEvent:
    max_epochs: int
    total_updates: int


@dataclass(frozen=True)
class TrainEndEvent:
    reason: Literal["completed", "max_updates", "stopped", "error"]
    update: int
    epoch: int


@dataclass(frozen=True)
class EvaluationResult:
    example_count: int
    loss: float | None
    accuracy: float | None
    unit: Literal["example", "token", "sequence"] = "example"
    unit_count: int | None = None
    perplexity: float | None = None
    exact_match_accuracy: float | None = None
    token_accuracy: float | None = None


@dataclass(frozen=True)
class TrainingWindowEvent:
    start_update: int
    end_update: int
    update_count: int
    closed_by: Literal["probe", "epoch_end", "terminal"]
    train_wall_time_ns: int
    eval_wall_time_ns: int | None
    train_device_time_ns: int | None = None
    eval_device_time_ns: int | None = None


class TrainerCallback(Protocol):
    def on_batch_end(self, *, step: int) -> None: ...
    def on_interval(self, *, metrics: dict[str, float]) -> None: ...
    def on_epoch_end(self, *, epoch: int, metrics: dict[str, float]) -> None: ...


class TrainerEventReceiver(Protocol):
    def on_update(self, event: UpdateEvent) -> None: ...
    def on_epoch(self, event: EpochEvent) -> None: ...
    def on_train_end(self, event: TrainEndEvent) -> None: ...


class SourceObjectiveReceiver(Protocol):
    def on_source_objective(self, event: SourceObjectiveSample) -> None: ...


class NullTrainerCallback:
    def on_batch_end(self, *, step: int) -> None:
        pass

    def on_interval(self, *, metrics: dict[str, float]) -> None:
        pass

    def on_epoch_end(self, *, epoch: int, metrics: dict[str, float]) -> None:
        pass


class NullTrainerEventReceiver:
    def on_update(self, event: UpdateEvent) -> None:
        pass

    def on_epoch(self, event: EpochEvent) -> None:
        pass

    def on_train_end(self, event: TrainEndEvent) -> None:
        pass
