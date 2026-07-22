"""Trainer for models that own their objective and expose ``forward(x, t)``."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from mlprosection import Tensor
from mlprosection.nn.layers import Layer
from mlprosection.optim import Optimizer
from mlprosection.profiling import ProfilingConfig

from .base import Trainer
from ..callbacks import TrainerCallback


@dataclass
class InternalLossHistory:
    """Losses emitted by a model whose forward pass already computes loss."""

    interval_loss: list[float] = field(default_factory=list)
    epoch_loss: list[float] = field(default_factory=list)


class InternalLossTrainer(Trainer):
    """Shared shuffled-batch loop for models with an internal loss layer.

    This keeps algorithm-specific models small: a model only implements
    ``forward(inputs, targets)`` and ``backward()``, while clipping, optimizer
    updates, callbacks, resumption state, and epoch accounting live here.
    """

    def __init__(
        self,
        model: Layer,
        optimizer: Optimizer,
        *,
        max_epoch: int,
        max_updates: int | None = None,
        batch_size: int,
        log_interval: int = 20,
        max_grad: float | None = None,
        drop_last: bool = True,
        callbacks: Iterable[TrainerCallback] | None = None,
        on_epoch_checkpoint: Callable[[int], None] | None = None,
        profiling_config: ProfilingConfig | None = None,
    ) -> None:
        super().__init__(
            model=model,
            criterion=None,
            optimizer=optimizer,
            max_epoch=max_epoch,
            max_updates=max_updates,
            batch_size=batch_size,
            log_interval=log_interval,
            drop_last=drop_last,
            profiling_config=profiling_config,
            callbacks=callbacks,
        )
        self.max_grad = max_grad
        self.on_epoch_checkpoint = on_epoch_checkpoint
        self.epoch = 0
        self.history = InternalLossHistory()

    def fit(self, xs: Tensor, ts: Tensor) -> InternalLossHistory:
        if len(xs) != len(ts):
            raise ValueError("inputs and targets must have the same sample count")
        if self.num_batches(len(xs)) == 0:
            raise ValueError("dataset is smaller than one training batch")

        self.start_time = time.time()
        self.train = True
        self.start_profiling_run()
        try:
            for epoch_index in range(self.epoch, self.max_epoch):
                if self.max_updates is not None and self.global_step >= self.max_updates:
                    break
                self.epoch = epoch_index + 1
                self.model.train(True)
                order = self.backend.xp.random.permutation(len(xs))
                shuffled_x, shuffled_t = xs[order], ts[order]
                epoch_total = 0.0
                epoch_samples = 0
                interval_total = 0.0
                interval_samples = 0
                started_ns = self.begin_profiled_epoch(split="train", epoch_index=epoch_index)

                for iteration, (batch_x, batch_t) in enumerate(self.iter_batches(shuffled_x, shuffled_t), start=1):
                    loss = self.internal_loss_step(batch_x, batch_t)

                    self.global_step += 1
                    batch_samples = len(batch_x)
                    loss_value = float(loss.data)
                    epoch_total += loss_value * batch_samples
                    epoch_samples += batch_samples
                    interval_total += loss_value * batch_samples
                    interval_samples += batch_samples
                    self._emit_batch_end()

                    if iteration % self.log_interval == 0 or iteration == self.num_batches(len(xs)):
                        average = interval_total / interval_samples
                        self.history.interval_loss.append(average)
                        self.losses.train.append(average)
                        log = {
                            "epoch": self.epoch,
                            "iteration": iteration,
                            "global_step": self.global_step,
                            "loss": average,
                            "elapsed_time": time.time() - self.start_time,
                        }
                        self.logs.train.append(log)
                        self._emit_interval(log)
                        interval_total = 0.0
                        interval_samples = 0

                    if self.max_updates is not None and self.global_step >= self.max_updates:
                        break

                epoch_loss = epoch_total / epoch_samples
                self.history.epoch_loss.append(epoch_loss)
                self.emit_epoch_metrics(epoch=self.epoch, metrics={"train/loss": epoch_loss})
                self.finish_profiled_epoch(
                    split="train", epoch_index=epoch_index, sample_count=epoch_samples,
                    started_ns=started_ns,
                )
                if self.on_epoch_checkpoint is not None:
                    self.on_epoch_checkpoint(self.epoch)
                if self.max_updates is not None and self.global_step >= self.max_updates:
                    break
        finally:
            self.finish_profiling_run()
        return self.history

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state["history"] = self.history.__dict__
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        super().load_state_dict(state)
        history = state.get("history", {})
        if isinstance(history, dict):
            self.history = InternalLossHistory(
                **{key: list(value) for key, value in history.items()}
            )
