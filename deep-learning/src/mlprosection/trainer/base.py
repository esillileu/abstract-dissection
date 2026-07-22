"""Minimal shared state for event-based trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlprosection.nn.types import Criterion, Layer
    from mlprosection.optim import Optimizer


class Trainer(ABC):
    """Own model/update counters only; executors own experiment policy and I/O."""

    def __init__(self, *, model: Layer, criterion: Criterion, optimizer: Optimizer) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.global_step = 0
        self.epoch = 0

    @property
    def backend(self):
        return self.model.backend

    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def state_dict(self) -> dict[str, int]:
        return {"global_step": self.global_step, "epoch": self.epoch}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.global_step = int(state["global_step"])
        self.epoch = int(state.get("epoch", 0))
