"""Truncated-BPTT trainer shared by language-model experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from mlprosection import Tensor
from mlprosection.nn.layers import Layer
from mlprosection.optim import Optimizer
from mlprosection.optim.transform import ClipGradNorm

from .callbacks import TrainerCallback


@dataclass
class TimeTrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    valid_loss: list[float] = field(default_factory=list)
    train_ppl: list[float] = field(default_factory=list)
    valid_ppl: list[float] = field(default_factory=list)


class TimeTrainer:
    """Train models exposing ``forward(xs, ts)``/``backward()`` over BPTT batches."""

    def __init__(
        self,
        model: Layer,
        optimizer: Optimizer,
        *,
        max_epoch: int,
        batch_size: int,
        time_size: int,
        log_interval: int = 20,
        max_grad: float | None = None,
        callbacks: Iterable[TrainerCallback] | None = None,
        on_epoch_end: Callable[[int], None] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.time_size = time_size
        self.log_interval = log_interval
        self.max_grad = max_grad
        self.callbacks = tuple(callbacks or ())
        self.on_epoch_end = on_epoch_end
        self.epoch = 0
        self.global_step = 0
        self.time_index = 0
        self.history = TimeTrainingHistory()

    @property
    def backend(self):
        return self.model.backend

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
        data_size = len(xs)
        max_iters = data_size // (self.batch_size * self.time_size)
        if max_iters < 1:
            raise ValueError("sequence is too short for batch_size * time_size")
        for epoch in range(self.epoch, self.max_epoch):
            self.epoch = epoch + 1
            self.model.train(True)
            total_loss = 0.0
            count = 0
            for iteration in range(max_iters):
                batch_x, batch_t = self.batch(xs, ts)
                loss = self.model.forward(batch_x, batch_t)
                self.model.backward()
                if self.max_grad is not None:
                    ClipGradNorm(self.max_grad)(list(self.model.named_parameters()))
                self.optimizer.update()
                self.global_step += 1
                total_loss += float(loss.data)
                count += 1
                for callback in self.callbacks:
                    callback.on_batch_end(step=self.global_step)
                if (iteration + 1) % self.log_interval == 0 or iteration + 1 == max_iters:
                    average = total_loss / count
                    ppl = float(self.backend.xp.exp(average))
                    self.history.train_loss.append(average)
                    self.history.train_ppl.append(ppl)
                    metrics = {"epoch": float(self.epoch), "iteration": float(self.global_step), "loss": average, "perplexity": ppl, "elapsed_time": time.time()}
                    for callback in self.callbacks:
                        callback.on_interval(metrics=metrics)
                    total_loss = 0.0
                    count = 0
            for callback in self.callbacks:
                callback.on_epoch_end(epoch=self.epoch, metrics={"train/perplexity": self.history.train_ppl[-1]})
            if self.on_epoch_end is not None:
                self.on_epoch_end(self.epoch)
        return self.history

    def evaluate_perplexity(self, xs: Tensor, ts: Tensor) -> float:
        reset = getattr(self.model, "reset_state", None)
        if reset is not None:
            reset()
        self.model.eval()
        total = 0.0
        count = 0
        data_size = len(xs)
        for start in range(0, data_size - self.time_size, self.time_size):
            batch_x = Tensor(xs.data[start:start + self.time_size][None, :], backend=self.backend)
            batch_t = Tensor(ts.data[start:start + self.time_size][None, :], backend=self.backend)
            loss = self.model.forward(batch_x, batch_t)
            total += float(loss.data)
            count += 1
        self.model.train(True)
        return float(self.backend.xp.exp(total / max(count, 1)))

    def state_dict(self) -> dict[str, object]:
        return {"epoch": self.epoch, "global_step": self.global_step, "time_index": self.time_index, "history": self.history.__dict__}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        self.time_index = int(state.get("time_index", 0))
        history = state.get("history", {})
        if isinstance(history, dict):
            self.history = TimeTrainingHistory(**{key: list(value) for key, value in history.items()})
