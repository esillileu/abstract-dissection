"""Shared event lifecycle for models that own their training objective."""

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
from mlprosection.nn.layers import Layer

from .base import Trainer


class InternalObjectiveTrainer(Trainer):
    """Base for trainers whose model implements ``forward(inputs, targets)``.

    It deliberately mirrors ForwardTrainer's event/counter contract without
    inheriting criterion-specific update behavior.
    """

    def __init__(
        self,
        model,
        optimizer,
        *,
        max_epochs: int,
        max_updates: int | None = None,
        event_receivers: Iterable[TrainerEventReceiver] | None = None,
    ) -> None:
        super().__init__(model=model, criterion=None, optimizer=optimizer)  # type: ignore[arg-type]
        if max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if max_updates is not None and max_updates < 1:
            raise ValueError("max_updates must be positive")
        self.max_epochs = max_epochs
        self.max_updates = max_updates
        self.event_receivers = tuple(event_receivers or ())

    def _at_update_limit(self) -> bool:
        return self.max_updates is not None and self.global_step >= self.max_updates

    def _learning_rate(self) -> float | tuple[float, ...]:
        value = getattr(self.optimizer, "lr", None)
        if value is None:
            raise RuntimeError("optimizer must expose the learning rate used for an update")
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

    def _emit_train_end(self, reason: Literal["completed", "max_updates", "stopped", "error"]) -> None:
        event = TrainEndEvent(reason=reason, update=self.global_step, epoch=self.epoch)
        for receiver in self.event_receivers:
            receiver.on_train_end(event)

    def _snapshot_recurrent_state(self) -> list[tuple[object, object, object]]:
        """Copy stateful time-layer values without assuming a model class."""
        snapshot: list[tuple[object, object, object]] = []
        seen: set[int] = set()

        def walk(value) -> None:
            if id(value) in seen:
                return
            if isinstance(value, (list, tuple)):
                seen.add(id(value))
                for item in value:
                    walk(item)
                return
            if not isinstance(value, Layer):
                return
            seen.add(id(value))
            if hasattr(value, "h") or hasattr(value, "c"):
                h = getattr(value, "h", None)
                c = getattr(value, "c", None)
                snapshot.append((value, None if h is None else h.copy(), None if c is None else c.copy()))
            for item in vars(value).values():
                walk(item)

        walk(self.model)
        return snapshot

    @staticmethod
    def _restore_recurrent_state(snapshot: list[tuple[object, object, object]]) -> None:
        for layer, h, c in snapshot:
            if hasattr(layer, "h"):
                layer.h = h
            if hasattr(layer, "c"):
                layer.c = c
