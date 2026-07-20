"""Truncated-BPTT specialization of the shared trainer base."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from mlprosection import Tensor
from mlprosection.nn.layers import Layer
from mlprosection.optim import Optimizer
from mlprosection.profiling import ProfilingConfig

from .base import Trainer
from .callbacks import TrainerCallback


@dataclass
class TimeTrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    valid_loss: list[float] = field(default_factory=list)
    train_ppl: list[float] = field(default_factory=list)
    valid_ppl: list[float] = field(default_factory=list)


class TimeTrainer(Trainer):
    """Shared trainer state plus BPTT batching and perplexity evaluation."""

    def __init__(
        self,
        model: Layer,
        optimizer: Optimizer,
        *,
        max_epoch: int,
        max_updates: int | None = None,
        batch_size: int,
        time_size: int,
        log_interval: int = 20,
        max_grad: float | None = None,
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
            profiling_config=profiling_config,
            callbacks=callbacks,
        )
        self.time_size = time_size
        self.max_grad = max_grad
        self.on_epoch_checkpoint = on_epoch_checkpoint
        self.epoch = 0
        self.time_index = 0
        self.history = TimeTrainingHistory()

    def batch(self, xs: Tensor, ts: Tensor) -> tuple[Tensor, Tensor]:
        xp = self.backend.xp
        data_size = len(xs)
        jump = data_size // self.batch_size
        batch_x = xp.empty((self.batch_size, self.time_size), dtype=xp.int64)
        batch_t = xp.empty((self.batch_size, self.time_size), dtype=xp.int64)
        for offset_index, offset in enumerate(range(0, jump * self.batch_size, jump)):
            positions = (xp.arange(self.time_size) + offset + self.time_index) % data_size
            batch_x[offset_index] = xs.data[positions]
            batch_t[offset_index] = ts.data[positions]
        self.time_index = (self.time_index + self.time_size) % data_size
        return Tensor(batch_x, backend=self.backend), Tensor(batch_t, backend=self.backend)

    def fit(self, xs: Tensor, ts: Tensor) -> TimeTrainingHistory:
        max_iters = len(xs) // (self.batch_size * self.time_size)
        if max_iters < 1:
            raise ValueError("sequence is too short for batch_size * time_size")
        self.start_time = time.time()
        for epoch in range(self.epoch, self.max_epoch):
            if self.max_updates is not None and self.global_step >= self.max_updates:
                break
            self.epoch = epoch + 1
            self.model.train(True)
            total_loss = 0.0
            count = 0
            for iteration in range(max_iters):
                batch_x, batch_t = self.batch(xs, ts)
                loss = self.model.forward(batch_x, batch_t)
                self.model.backward()
                self.clip_gradients()
                self.optimizer.update()
                detach = getattr(self.model, "detach_state", None)
                if detach is not None:
                    detach()
                self.global_step += 1
                total_loss += float(loss.data)
                count += 1
                self._emit_batch_end()
                if (iteration + 1) % self.log_interval == 0 or iteration + 1 == max_iters:
                    average = total_loss / count
                    ppl = float(self.backend.xp.exp(average))
                    self.history.train_loss.append(average)
                    self.history.train_ppl.append(ppl)
                    self.losses.train.append(average)
                    log = {"epoch": float(self.epoch), "iteration": float(self.global_step), "loss": average, "perplexity": ppl, "elapsed_time": time.time() - self.start_time}
                    self.logs.train.append(log)
                    self._emit_interval(log)
                    total_loss = 0.0
                    count = 0
                if self.max_updates is not None and self.global_step >= self.max_updates:
                    break
            self.emit_epoch_metrics(epoch=self.epoch, metrics={"train/perplexity": self.history.train_ppl[-1]})
            if self.on_epoch_checkpoint is not None:
                self.on_epoch_checkpoint(self.epoch)
            if self.max_updates is not None and self.global_step >= self.max_updates:
                break
        return self.history

    def evaluate_perplexity(self, xs: Tensor, ts: Tensor) -> float:
        reset = getattr(self.model, "reset_state", None)
        if reset is not None:
            reset()
        self.model.eval()
        total = 0.0
        count = 0
        for start in range(0, len(xs) - self.time_size, self.time_size):
            batch_x = Tensor(xs.data[start:start + self.time_size][None, :], backend=self.backend)
            batch_t = Tensor(ts.data[start:start + self.time_size][None, :], backend=self.backend)
            total += float(self.model.forward(batch_x, batch_t).data)
            count += 1
        self.model.train(True)
        if reset is not None:
            reset()
        return float(self.backend.xp.exp(total / max(count, 1)))

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state.update({"time_index": self.time_index, "history": self.history.__dict__})
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        super().load_state_dict(state)
        self.time_index = int(state.get("time_index", 0))
        history = state.get("history", {})
        if isinstance(history, dict):
            self.history = TimeTrainingHistory(**{key: list(value) for key, value in history.items()})
