"""Event-based truncated-BPTT trainer for the book language-model runs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from mlprosection import Tensor
from mlprosection.events import EpochEvent, EvaluationResult, SourceObjectiveSample, UpdateEvent
from .event import EventTrainer


class LanguageModelTrainer(EventTrainer):
    def __init__(
        self,
        model,
        objective,
        optimizer,
        *,
        max_epochs: int,
        batch_size: int,
        time_size: int,
        max_updates: int | None = None,
        epoch_cursor: str = "continuous",
        epoch_recurrent_state: str = "continuous",
        evaluator_batch_size: int = 10,
        evaluator_time_size: int = 35,
        evaluator_drop_remainder: bool = True,
        event_receivers=None,
    ) -> None:
        super().__init__(model, objective, optimizer, max_epochs=max_epochs, max_updates=max_updates, event_receivers=event_receivers)
        if batch_size < 1 or time_size < 1:
            raise ValueError("batch_size and time_size must be positive")
        self.batch_size = batch_size
        self.time_size = time_size
        self.time_index = 0
        self.iteration_in_epoch = 0
        self._batch_offsets = None
        self._time_offsets = None
        self._batch_data_size: int | None = None
        if epoch_cursor not in {"continuous", "reset"}:
            raise ValueError("epoch_cursor must be continuous or reset")
        if epoch_recurrent_state not in {"continuous", "reset"}:
            raise ValueError("epoch_recurrent_state must be continuous or reset")
        self.epoch_cursor = epoch_cursor
        self.epoch_recurrent_state = epoch_recurrent_state
        self.evaluator_batch_size = evaluator_batch_size
        self.evaluator_time_size = evaluator_time_size
        self.evaluator_drop_remainder = evaluator_drop_remainder

    def fit(self, xs: Tensor, ts: Tensor) -> None:
        if len(xs) != len(ts):
            raise ValueError("inputs and targets must have the same token count")
        max_iterations = len(xs) // (self.batch_size * self.time_size)
        if max_iterations < 1:
            raise ValueError("sequence is too short for batch_size * time_size")
        reason: Literal["completed", "max_updates", "stopped", "error"] = "completed"
        try:
            for epoch_index in range(self.epoch, self.max_epochs):
                if self._at_update_limit():
                    reason = "max_updates"
                    break
                self.epoch = epoch_index + 1
                if self.epoch_cursor == "reset":
                    self.time_index = 0
                    self.iteration_in_epoch = 0
                if self.epoch_recurrent_state == "reset":
                    self.model.reset_runtime_state()
                start_update = self.global_step + 1
                for iteration in range(self.iteration_in_epoch, max_iterations):
                    batch_x, batch_t = self._batch(xs, ts)
                    self.model.train(True)
                    prediction = self.model.forward(batch_x)
                    result = self.objective.forward(prediction, batch_t)
                    self.model.backward(self.objective.backward())
                    self.optimizer.update()
                    self.model.detach_runtime_state()
                    self.global_step += 1
                    self.iteration_in_epoch = iteration + 1
                    self._emit_update(UpdateEvent(
                        update=self.global_step, epoch=self.epoch, batch_size=self.batch_size,
                        loss=result.loss, learning_rate=self._learning_rate(),
                    ))
                    self._emit_source_objective(SourceObjectiveSample(
                        update=self.global_step, epoch=self.epoch, local_iteration=iteration,
                        objective=result.loss, unit_count=result.unit_count,
                    ))
                    if self._at_update_limit():
                        reason = "max_updates"
                        break
                self._emit_epoch(EpochEvent(
                    epoch=self.epoch, start_update=start_update, end_update=self.global_step,
                    sample_count=(self.iteration_in_epoch * self.batch_size * self.time_size),
                ))
                self.iteration_in_epoch = 0
                if self._at_update_limit():
                    break
        except BaseException:
            reason = "error"
            raise
        finally:
            self._emit_train_end(reason)

    def planned_total_updates(self, token_count: int) -> int:
        updates_per_epoch = token_count // (self.batch_size * self.time_size)
        epoch_total = updates_per_epoch * self.max_epochs
        return epoch_total if self.max_updates is None else min(epoch_total, self.max_updates)

    def evaluate(
        self,
        xs: Tensor,
        ts: Tensor,
        *,
        metrics: Sequence[Literal["perplexity"]] = ("perplexity",),
    ) -> EvaluationResult:
        if tuple(metrics) != ("perplexity",):
            raise ValueError("LanguageModelTrainer supports only perplexity evaluation")
        if len(xs) != len(ts):
            raise ValueError("inputs and targets must have the same token count")
        if not len(xs):
            raise ValueError("evaluation corpus must not be empty")
        saved_state = self._snapshot_evaluation_state()
        xp = self.backend.xp
        total_nll = xp.asarray(0.0, dtype=xp.float64)
        token_count = 0
        self.model.train(False)
        self.objective.train(False)
        self.model.reset_runtime_state()
        try:
            # Keep the requested ten parallel streams for PTB, while allowing
            # tiny test corpora to retain at least one full time window.
            batch_size = min(self.evaluator_batch_size, len(xs))
            eval_time_size = min(
                self.evaluator_time_size,
                max(1, len(xs) // batch_size),
            )
            if batch_size < 1:
                raise ValueError("evaluator_batch_size must be positive")
            stream_length = len(xs) // batch_size
            if not self.evaluator_drop_remainder and len(xs) % batch_size:
                stream_length += 1
            if stream_length < 1:
                raise ValueError("evaluation corpus is too short for evaluator_batch_size")
            usable = stream_length * batch_size
            if usable > len(xs):
                positions = self.backend.xp.arange(usable) % len(xs)
                flat_x, flat_t = xs.data[positions], ts.data[positions]
            else:
                flat_x, flat_t = xs.data[:usable], ts.data[:usable]
            stream_x = flat_x.reshape(batch_size, stream_length)
            stream_t = flat_t.reshape(batch_size, stream_length)
            for start in range(0, stream_length, eval_time_size):
                end = min(start + eval_time_size, stream_length)
                if self.evaluator_drop_remainder and end - start < eval_time_size:
                    break
                batch_x = Tensor(stream_x[:, start:end], backend=self.backend)
                batch_t = Tensor(stream_t[:, start:end], backend=self.backend)
                prediction = self.model.forward(batch_x, cache=False)
                result = self.objective.forward(prediction, batch_t, cache=False)
                count = (end - start) * batch_size
                total_nll = total_nll + result.loss.data.astype(xp.float64) * count
                token_count += count
        finally:
            self._restore_evaluation_state(saved_state)
        mean_nll = total_nll / token_count
        host_metrics = self.backend.to_numpy(xp.stack((mean_nll, xp.exp(mean_nll))))
        return EvaluationResult(
            example_count=token_count, loss=float(host_metrics[0]), accuracy=None,
            unit="token", unit_count=token_count,
            perplexity=float(host_metrics[1]),
        )

    def _batch(self, xs: Tensor, ts: Tensor) -> tuple[Tensor, Tensor]:
        xp = self.backend.xp
        data_size = len(xs)
        if self._batch_offsets is None or self._batch_data_size != data_size:
            jump = data_size // self.batch_size
            self._batch_offsets = xp.arange(self.batch_size) * jump
            self._time_offsets = xp.arange(self.time_size)
            self._batch_data_size = data_size
        positions = (
            self._batch_offsets[:, None]
            + self.time_index
            + self._time_offsets[None, :]
        ) % data_size
        self.time_index = (self.time_index + self.time_size) % data_size
        return (
            Tensor(xs.data[positions], backend=self.backend),
            Tensor(ts.data[positions], backend=self.backend),
        )

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state.update({"time_index": self.time_index, "iteration_in_epoch": self.iteration_in_epoch})
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        super().load_state_dict(state)
        self.time_index = int(state.get("time_index", 0))
        self.iteration_in_epoch = int(state.get("iteration_in_epoch", 0))
