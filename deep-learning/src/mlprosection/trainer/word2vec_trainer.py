"""Event-based trainer for full-softmax and negative-sampling Word2Vec."""

from __future__ import annotations

from typing import Literal

from mlprosection import Tensor
from mlprosection.events import EpochEvent, SourceObjectiveSample, UpdateEvent
from .event import EventTrainer


class Word2VecTrainer(EventTrainer):
    def __init__(
        self,
        model,
        objective,
        optimizer,
        *,
        batch_adapter,
        max_epochs: int,
        batch_size: int,
        max_updates: int | None = None,
        drop_last: bool = True,
        event_receivers=None,
        batch_rng=None,
    ) -> None:
        super().__init__(
            model, objective, optimizer, max_epochs=max_epochs, max_updates=max_updates,
            event_receivers=event_receivers, batch_rng=batch_rng,
        )
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.batch_adapter = batch_adapter
        self.batch_cursor = 0

    def fit(self, contexts: Tensor, targets: Tensor) -> None:
        if len(contexts) != len(targets):
            raise ValueError("contexts and targets must have the same sample count")
        if self._num_batches(len(contexts)) == 0:
            raise ValueError("dataset is smaller than one training batch")
        reason: Literal["completed", "max_updates", "stopped", "error"] = "completed"
        try:
            for epoch_index in range(self.epoch, self.max_epochs):
                if self._at_update_limit():
                    reason = "max_updates"
                    break
                self.epoch = epoch_index + 1
                start_update, sample_count = self.global_step + 1, 0
                order = self.batch_rng.permutation(len(contexts))
                shuffled_contexts, shuffled_targets = contexts[order], targets[order]
                for local_iteration, (batch_x, batch_t) in enumerate(self._iter_batches(shuffled_contexts, shuffled_targets)):
                    model_x, objective_t = self.batch_adapter.prepare(batch_x, batch_t)
                    objective_batch = self.objective.prepare(objective_t)
                    prediction = self.model.forward(
                        model_x,
                        candidates=objective_batch.candidates,
                    )
                    result = self.objective.forward(
                        prediction,
                        objective_batch.target,
                        replay_context=objective_batch.replay_context,
                        example_count=len(batch_x),
                    )
                    gradient = self.objective.backward()
                    self.model.backward(gradient)
                    self.optimizer.update()
                    probe_state = self._snapshot_evaluation_state()
                    try:
                        post_prediction = self.model.forward(
                            model_x,
                            candidates=objective_batch.candidates,
                            cache=False,
                        )
                        post_result = self.objective.forward(
                            post_prediction,
                            objective_batch.target,
                            cache=False,
                            replay_context=objective_batch.replay_context,
                            example_count=len(batch_x),
                        )
                    finally:
                        self._restore_evaluation_state(probe_state)
                    self.global_step += 1
                    sample_count += len(batch_x)
                    self._emit_update(UpdateEvent(
                        update=self.global_step, epoch=self.epoch, batch_size=len(batch_x),
                        loss=(
                            post_result.reporting_loss
                            if post_result.reporting_loss is not None
                            else post_result.loss
                        ),
                        learning_rate=self._learning_rate(),
                        book_loss=post_result.loss,
                    ))
                    self._emit_source_objective(SourceObjectiveSample(
                        update=self.global_step, epoch=self.epoch,
                        local_iteration=local_iteration,
                        objective=(
                            result.reporting_loss
                            if result.reporting_loss is not None
                            else result.loss
                        ),
                        unit_count=result.unit_count,
                        book_objective=result.loss,
                    ))
                    self.batch_cursor = local_iteration + 1
                    if self._at_update_limit():
                        reason = "max_updates"
                        break
                self._emit_epoch(EpochEvent(
                    epoch=self.epoch, start_update=start_update,
                    end_update=self.global_step, sample_count=sample_count,
                ))
                self.batch_cursor = 0
                if self._at_update_limit():
                    break
        except BaseException:
            reason = "error"
            raise
        finally:
            self._emit_train_end(reason)

    def _iter_batches(self, x: Tensor, t: Tensor):
        size = len(x) - (len(x) % self.batch_size) if self.drop_last else len(x)
        for start in range(0, size, self.batch_size):
            yield x[start:start + self.batch_size], t[start:start + self.batch_size]

    def _num_batches(self, size: int) -> int:
        return size // self.batch_size if self.drop_last else (size + self.batch_size - 1) // self.batch_size

    def planned_total_updates(self, sample_count: int) -> int:
        epoch_total = self._num_batches(sample_count) * self.max_epochs
        return epoch_total if self.max_updates is None else min(epoch_total, self.max_updates)

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state["batch_cursor"] = self.batch_cursor
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        super().load_state_dict(state)
        self.batch_cursor = int(state.get("batch_cursor", 0))
