"""Minimal shared state for event-based trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepscratch.nn.model import Model
    from deepscratch.nn.objective import Objective
    from deepscratch.optim import Optimizer


class Trainer(ABC):
    """Own model/update counters only; executors own experiment policy and I/O."""

    def __init__(
        self,
        *,
        model: Model,
        objective: Objective,
        optimizer: Optimizer,
        batch_rng=None,
    ) -> None:
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.global_step = 0
        self.epoch = 0
        self.batch_rng = batch_rng or model.backend.random_stream("batch_order")

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

    def _snapshot_evaluation_state(self):
        rng_state = self.backend.random_stream_states()
        return (
            self.model.training,
            self.objective.training,
            self.model.snapshot_runtime_state(),
            rng_state,
        )

    def _restore_evaluation_state(self, state) -> None:
        model_mode, objective_mode, runtime_state, rng_state = state
        self.model.restore_runtime_state(runtime_state)
        self.model.train(model_mode)
        self.objective.train(objective_mode)
        self.backend.restore_random_stream_states(rng_state)


BaseTrainer = Trainer
