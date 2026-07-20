"""Small scalar-only hooks emitted by trainers."""

from __future__ import annotations

from typing import Protocol


class TrainerCallback(Protocol):
    def on_batch_end(self, *, step: int) -> None: ...

    def on_interval(self, *, metrics: dict[str, float]) -> None: ...

    def on_epoch_end(self, *, epoch: int, metrics: dict[str, float]) -> None: ...


class NullTrainerCallback:
    def on_batch_end(self, *, step: int) -> None:
        pass

    def on_interval(self, *, metrics: dict[str, float]) -> None:
        pass

    def on_epoch_end(self, *, epoch: int, metrics: dict[str, float]) -> None:
        pass
