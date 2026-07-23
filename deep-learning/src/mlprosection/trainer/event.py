from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from mlprosection.events import (
    EpochEvent,
    SourceObjectiveSample,
    TrainEndEvent,
    TrainerEventReceiver,
    UpdateEvent,
)

from .base import Trainer


class EventTrainer(Trainer):
    def __init__(
        self, model, objective, optimizer, *, max_epochs: int,
        max_updates: int | None = None,
        event_receivers: Iterable[TrainerEventReceiver] | None = None,
    ) -> None:
        super().__init__(model=model, objective=objective, optimizer=optimizer)
        if max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if max_updates is not None and max_updates < 1:
            raise ValueError("max_updates must be positive")
        self.max_epochs = max_epochs
        self.max_updates = max_updates
        self.event_receivers = tuple(event_receivers or ())

    def _at_update_limit(self) -> bool:
        return self.max_updates is not None and self.global_step >= self.max_updates

    def _learning_rate(self) -> float:
        value = getattr(self.optimizer, "lr", None)
        if value is None:
            raise RuntimeError("optimizer must expose the learning rate")
        return float(value)

    def _emit_update(self, event: UpdateEvent) -> None:
        for receiver in self.event_receivers:
            receiver.on_update(event)

    def _emit_source_objective(self, event: SourceObjectiveSample) -> None:
        for receiver in self.event_receivers:
            callback = getattr(receiver, "on_source_objective", None)
            if callback is not None:
                callback(event)

    def _emit_epoch(self, event: EpochEvent) -> None:
        for receiver in self.event_receivers:
            receiver.on_epoch(event)

    def _emit_train_end(
        self, reason: Literal["completed", "max_updates", "stopped", "error"]
    ) -> None:
        event = TrainEndEvent(
            reason=reason, update=self.global_step, epoch=self.epoch
        )
        for receiver in self.event_receivers:
            receiver.on_train_end(event)
