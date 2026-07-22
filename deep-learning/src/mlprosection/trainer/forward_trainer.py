"""Event-based trainer for supervised forward/criterion/backward updates."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal

from mlprosection.events import (
    EpochEvent,
    EvaluationResult,
    TrainEndEvent,
    TrainerEventReceiver,
    UpdateEvent,
)

from .base import Trainer
from .utils import clip_grads

if TYPE_CHECKING:
    from mlprosection import Tensor
    from mlprosection.nn.types import Criterion, Layer
    from mlprosection.optim import Optimizer


class ForwardTrainer(Trainer):
    """Run supervised updates and emit facts; never own experiment policy or I/O."""

    def __init__(
        self,
        model: Layer,
        criterion: Criterion,
        optimizer: Optimizer,
        *,
        max_epochs: int,
        batch_size: int,
        max_updates: int | None = None,
        drop_last: bool = False,
        sampling_method: Literal["permutation_per_epoch", "with_replacement"] = "permutation_per_epoch",
        max_grad: float | None = None,
        event_receivers: Iterable[TrainerEventReceiver] | None = None,
    ) -> None:
        super().__init__(model=model, criterion=criterion, optimizer=optimizer)
        if max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_updates is not None and max_updates < 1:
            raise ValueError("max_updates must be positive")
        self.max_epochs = max_epochs
        self.max_updates = max_updates
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampling_method = sampling_method
        self.max_grad = max_grad
        self.event_receivers = tuple(event_receivers or ())

    def fit(self, x_train: Tensor, t_train: Tensor) -> None:
        if len(x_train) != len(t_train):
            raise ValueError("inputs and targets must have the same sample count")
        if self._num_batches(len(x_train)) == 0:
            raise ValueError("dataset is smaller than one training batch")

        reason: Literal["completed", "max_updates", "stopped", "error"] = "completed"
        try:
            for epoch_index in range(self.epoch, self.max_epochs):
                if self._at_update_limit():
                    reason = "max_updates"
                    break
                self.epoch = epoch_index + 1
                start_update = self.global_step + 1
                sample_count = 0
                shuffled_x, shuffled_t = self._sample_epoch(x_train, t_train)
                for batch_x, batch_t in self._iter_batches(shuffled_x, shuffled_t):
                    loss = self._update(batch_x, batch_t)
                    self.global_step += 1
                    sample_count += len(batch_x)
                    self._emit_update(
                        UpdateEvent(
                            update=self.global_step,
                            epoch=self.epoch,
                            batch_size=len(batch_x),
                            loss=loss,
                            learning_rate=self._learning_rate(),
                        )
                    )
                    if self._at_update_limit():
                        reason = "max_updates"
                        break
                self._emit_epoch(
                    EpochEvent(
                        epoch=self.epoch,
                        start_update=start_update,
                        end_update=self.global_step,
                        sample_count=sample_count,
                    )
                )
                if self._at_update_limit():
                    break
        except BaseException:
            reason = "error"
            raise
        finally:
            self._emit_train_end(
                TrainEndEvent(reason=reason, update=self.global_step, epoch=self.epoch)
            )

    def evaluate(
        self,
        x: Tensor,
        t: Tensor,
        *,
        metrics: Sequence[Literal["loss", "accuracy"]] = ("loss", "accuracy"),
    ) -> EvaluationResult:
        """Evaluate an executor-selected fixed source without consuming train RNG."""
        if len(x) != len(t):
            raise ValueError("inputs and targets must have the same sample count")
        if not metrics:
            raise ValueError("at least one evaluation metric is required")
        unknown = set(metrics) - {"loss", "accuracy"}
        if unknown:
            raise ValueError(f"unsupported evaluation metrics: {sorted(unknown)}")
        if self._num_batches(len(x)) == 0:
            raise ValueError("evaluation source is smaller than one batch")

        xp = self.backend.xp
        was_training = bool(getattr(self.model, "training", True))
        total_loss = xp.asarray(0.0, dtype=x.data.dtype)
        total_correct = xp.asarray(0, dtype=xp.int64)
        sample_count = 0
        self.model.train(False)
        try:
            for batch_x, batch_t in self._iter_batches(x, t):
                y = self.model.forward(batch_x)
                if "loss" in metrics:
                    loss = self.criterion.forward(y, batch_t)
                    total_loss += loss.data * len(batch_x)
                if "accuracy" in metrics:
                    total_correct += self._correct_count(y, batch_t)
                sample_count += len(batch_x)
        finally:
            self.model.train(was_training)
        return EvaluationResult(
            example_count=sample_count,
            loss=(self.backend.scalar_to_float(total_loss) / sample_count)
            if "loss" in metrics else None,
            accuracy=(self.backend.scalar_to_int(total_correct) / sample_count)
            if "accuracy" in metrics else None,
        )

    def _update(self, x: Tensor, t: Tensor) -> Tensor:
        self.model.train(True)
        y = self.model.forward(x)
        self.criterion.forward(y, t)
        dx = self.criterion.backward()
        self.model.backward(dx)
        if self.max_grad is not None:
            clip_grads(list(self.model.named_parameters()), self.max_grad)
        self.optimizer.update()
        # DS1's canonical update record is the mean objective after the update.
        return self.criterion.forward(self.model.forward(x), t)

    def _sample_epoch(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        xp = self.backend.xp
        if self.sampling_method == "permutation_per_epoch":
            indices = xp.random.permutation(len(x))
        elif self.sampling_method == "with_replacement":
            updates = len(x) // self.batch_size
            if updates == 0:
                raise ValueError("dataset is smaller than one training batch")
            indices = xp.random.randint(0, len(x), size=updates * self.batch_size)
        else:
            raise ValueError(f"unsupported sampling_method: {self.sampling_method}")
        return x[indices], t[indices]

    def _iter_batches(self, x: Tensor, t: Tensor):
        size = len(x)
        if self.drop_last:
            size -= size % self.batch_size
        for start in range(0, size, self.batch_size):
            yield x[start:start + self.batch_size], t[start:start + self.batch_size]

    def _num_batches(self, size: int) -> int:
        return size // self.batch_size if self.drop_last else (size + self.batch_size - 1) // self.batch_size

    def _correct_count(self, y: Tensor, t: Tensor):
        xp = self.backend.xp
        y_data, t_data = y.data, t.data
        labels = t_data.argmax(axis=1) if t_data.size == y_data.size and t_data.ndim > 1 else t_data.reshape(-1)
        predictions = y_data.argmax(axis=1) if y_data.ndim > 1 and y_data.shape[1] > 1 else (y_data.reshape(-1) >= 0.5).astype(labels.dtype)
        return xp.sum(predictions == labels)

    def _learning_rate(self) -> float | tuple[float, ...]:
        value = getattr(self.optimizer, "lr", None)
        if value is None:
            raise RuntimeError("optimizer must expose the learning rate used for an update")
        return float(value)

    def _at_update_limit(self) -> bool:
        return self.max_updates is not None and self.global_step >= self.max_updates

    def _emit_update(self, event: UpdateEvent) -> None:
        for receiver in self.event_receivers:
            receiver.on_update(event)

    def _emit_epoch(self, event: EpochEvent) -> None:
        for receiver in self.event_receivers:
            receiver.on_epoch(event)

    def _emit_train_end(self, event: TrainEndEvent) -> None:
        for receiver in self.event_receivers:
            receiver.on_train_end(event)
