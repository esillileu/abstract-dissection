"""Event-based trainer and greedy evaluator for character Seq2seq models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from mlprosection import Tensor
from mlprosection.events import EpochEvent, EvaluationResult, UpdateEvent
from .internal_objective import InternalObjectiveTrainer


class Seq2seqTrainer(InternalObjectiveTrainer):
    def __init__(
        self,
        model,
        optimizer,
        *,
        max_epochs: int,
        batch_size: int,
        start_id: int,
        max_decode_steps: int | None = None,
        max_updates: int | None = None,
        drop_last: bool = False,
        event_receivers=None,
    ) -> None:
        super().__init__(model, optimizer, max_epochs=max_epochs, max_updates=max_updates, event_receivers=event_receivers)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.start_id = start_id
        self.max_decode_steps = max_decode_steps
        self.drop_last = drop_last
        self.batch_cursor = 0

    def fit(self, xs: Tensor, ts: Tensor) -> None:
        if len(xs) != len(ts):
            raise ValueError("inputs and targets must have the same sequence count")
        if self._num_batches(len(xs)) == 0:
            raise ValueError("dataset is smaller than one training batch")
        reason: Literal["completed", "max_updates", "stopped", "error"] = "completed"
        try:
            for epoch_index in range(self.epoch, self.max_epochs):
                if self._at_update_limit():
                    reason = "max_updates"
                    break
                self.epoch = epoch_index + 1
                start_update, sample_count = self.global_step + 1, 0
                order = self.backend.xp.random.permutation(len(xs))
                shuffled_x, shuffled_t = xs[order], ts[order]
                for iteration, (batch_x, batch_t) in enumerate(self._iter_batches(shuffled_x, shuffled_t)):
                    self.model.train(True)
                    self.model.forward(batch_x, batch_t)
                    self.model.backward()
                    self.optimizer.update()
                    post_loss = self.model.forward(batch_x, batch_t)
                    self.global_step += 1
                    sample_count += len(batch_x)
                    self.batch_cursor = iteration + 1
                    self._emit_update(UpdateEvent(
                        update=self.global_step, epoch=self.epoch, batch_size=len(batch_x),
                        loss=post_loss, learning_rate=self._learning_rate(),
                    ))
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

    def evaluate(
        self,
        xs: Tensor,
        ts: Tensor,
        *,
        metrics: Sequence[Literal["exact_match_accuracy", "token_accuracy"]] = ("exact_match_accuracy",),
    ) -> EvaluationResult:
        requested = set(metrics)
        if not requested or requested - {"exact_match_accuracy", "token_accuracy"}:
            raise ValueError("unsupported Seq2seq evaluation metrics")
        if len(xs) != len(ts):
            raise ValueError("inputs and targets must have the same sequence count")
        if not len(xs):
            raise ValueError("evaluation source must not be empty")
        was_training = bool(getattr(self.model, "training", True))
        saved_state = self._snapshot_recurrent_state()
        exact, token_correct, token_count = 0, 0, 0
        host_targets = self.backend.to_numpy(ts.data)
        self.model.train(False)
        try:
            for index in range(len(xs)):
                question = Tensor(xs.data[index:index + 1], backend=self.backend)
                answer = host_targets[index]
                expected = [int(value) for value in answer[1:]]
                sample_size = self.max_decode_steps if self.max_decode_steps is not None else len(expected)
                predicted = self.model.generate(question, self.start_id, sample_size)
                exact += int(predicted == expected)
                token_correct += sum(left == right for left, right in zip(predicted, expected, strict=True))
                token_count += len(expected)
        finally:
            self._restore_recurrent_state(saved_state)
            self.model.train(was_training)
        return EvaluationResult(
            example_count=len(xs), loss=None, accuracy=None,
            unit="sequence", unit_count=len(xs),
            exact_match_accuracy=(exact / len(xs)) if "exact_match_accuracy" in requested else None,
            token_accuracy=(token_correct / token_count) if "token_accuracy" in requested else None,
        )

    def _iter_batches(self, x: Tensor, t: Tensor):
        size = len(x) - (len(x) % self.batch_size) if self.drop_last else len(x)
        for start in range(0, size, self.batch_size):
            yield x[start:start + self.batch_size], t[start:start + self.batch_size]

    def _num_batches(self, size: int) -> int:
        return size // self.batch_size if self.drop_last else (size + self.batch_size - 1) // self.batch_size

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state["batch_cursor"] = self.batch_cursor
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        super().load_state_dict(state)
        self.batch_cursor = int(state.get("batch_cursor", 0))
