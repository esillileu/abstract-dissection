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
                start_update = self.global_step + 1
                for iteration in range(self.iteration_in_epoch, max_iterations):
                    batch_x, batch_t = self._batch(xs, ts)
                    state_before = self.model.snapshot_runtime_state()
                    self.model.train(True)
                    prediction = self.model.forward(batch_x)
                    result = self.objective.forward(prediction, batch_t)
                    self.model.backward(self.objective.backward())
                    self.optimizer.update()
                    probe_state = self._snapshot_evaluation_state()
                    self.model.restore_runtime_state(state_before)
                    try:
                        post_prediction = self.model.forward(batch_x, cache=False)
                        post_result = self.objective.forward(
                            post_prediction, batch_t, cache=False,
                            replay_context=result.replay_context,
                        )
                    finally:
                        self._restore_evaluation_state(probe_state)
                    self.model.detach_runtime_state()
                    self.global_step += 1
                    self.iteration_in_epoch = iteration + 1
                    self._emit_update(UpdateEvent(
                        update=self.global_step, epoch=self.epoch, batch_size=self.batch_size,
                        loss=post_result.loss, learning_rate=self._learning_rate(),
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
            for start in range(0, len(xs), self.time_size):
                end = min(start + self.time_size, len(xs))
                batch_x = Tensor(xs.data[start:end][None, :], backend=self.backend)
                batch_t = Tensor(ts.data[start:end][None, :], backend=self.backend)
                prediction = self.model.forward(batch_x, cache=False)
                result = self.objective.forward(prediction, batch_t, cache=False)
                count = end - start
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
