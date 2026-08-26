"""Event-based trainer and greedy evaluator for character Seq2seq models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from deepscratch.core import Tensor
from deepscratch.nn.model.base import GenerativeModel
from deepscratch.trainer.events import EpochEvent, EvaluationResult, UpdateEvent

from .event import EventTrainer


class Seq2seqTrainer(EventTrainer):
    def __init__(
        self,
        model,
        objective,
        optimizer,
        *,
        max_epochs: int,
        batch_size: int,
        start_id: int,
        eval_batch_size: int | None = None,
        max_decode_steps: int | None = None,
        max_updates: int | None = None,
        drop_last: bool = False,
        loss_timing: Literal["pre_update", "post_update"] = "post_update",
        event_receivers=None,
        batch_rng=None,
    ) -> None:
        super().__init__(
            model,
            objective,
            optimizer,
            max_epochs=max_epochs,
            max_updates=max_updates,
            event_receivers=event_receivers,
            batch_rng=batch_rng,
        )
        if not isinstance(model, GenerativeModel):
            raise TypeError("Seq2seqTrainer requires the GenerativeModel capability")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if eval_batch_size is not None and eval_batch_size < 1:
            raise ValueError("eval_batch_size must be positive")
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.start_id = start_id
        self.max_decode_steps = max_decode_steps
        self.drop_last = drop_last
        if loss_timing not in {"pre_update", "post_update"}:
            raise ValueError("loss_timing must be pre_update or post_update")
        self.loss_timing = loss_timing
        self.batch_cursor = 0
        self.samples_seen = 0
        self.last_predictions = None

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
                order = self.batch_rng.permutation(len(xs))
                shuffled_x, shuffled_t = xs[order], ts[order]
                for iteration, (batch_x, batch_t) in enumerate(
                    self._iter_batches(shuffled_x, shuffled_t)
                ):
                    self.model.train(True)
                    decoder_x, objective_t = batch_t[:, :-1], batch_t[:, 1:]
                    prediction = self.model.forward(batch_x, decoder_x)
                    result = self.objective.forward(prediction, objective_t)
                    self.model.backward(self.objective.backward())
                    self.optimizer.update()
                    if self.loss_timing == "post_update":
                        probe_state = self._snapshot_evaluation_state()
                        try:
                            post_prediction = self.model.forward(
                                batch_x, decoder_x, cache=False
                            )
                            post_result = self.objective.forward(
                                post_prediction,
                                objective_t,
                                cache=False,
                                replay_context=result.replay_context,
                            )
                        finally:
                            self._restore_evaluation_state(probe_state)
                        update_loss = post_result.loss
                    else:
                        update_loss = result.loss
                    self.global_step += 1
                    batch_samples = len(batch_x)
                    sample_count += batch_samples
                    self.samples_seen += batch_samples
                    self.batch_cursor = iteration + 1
                    self._emit_update(
                        UpdateEvent(
                            update=self.global_step,
                            epoch=self.epoch,
                            batch_size=len(batch_x),
                            loss=update_loss,
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
        metrics: Sequence[Literal["exact_match_accuracy", "token_accuracy"]] = (
            "exact_match_accuracy",
        ),
    ) -> EvaluationResult:
        requested = set(metrics)
        if not requested or requested - {"exact_match_accuracy", "token_accuracy"}:
            raise ValueError("unsupported Seq2seq evaluation metrics")
        if len(xs) != len(ts):
            raise ValueError("inputs and targets must have the same sequence count")
        if not len(xs):
            raise ValueError("evaluation source must not be empty")
        saved_state = self._snapshot_evaluation_state()
        exact, token_correct, token_count = 0, 0, 0
        device_predictions = []
        self.model.train(False)
        self.objective.train(False)
        expected_size = ts.shape[1] - 1
        sample_size = (
            self.max_decode_steps
            if self.max_decode_steps is not None
            else expected_size
        )
        if sample_size != expected_size:
            raise ValueError("max_decode_steps must match the fixed target length")
        try:
            for start in range(0, len(xs), self.eval_batch_size):
                question = Tensor(
                    xs.data[start : start + self.eval_batch_size],
                    backend=self.backend,
                )
                device_predictions.append(
                    self.model.generate_device(question, self.start_id, sample_size)
                )
        finally:
            self._restore_evaluation_state(saved_state)
        xp = self.backend.xp
        predictions = xp.concatenate(device_predictions, axis=0)
        paired = xp.stack((predictions, ts.data[:, 1:]))
        host_predictions, host_targets = self.backend.to_numpy(paired)
        self.last_predictions = host_predictions
        for predicted, expected in zip(host_predictions, host_targets, strict=True):
            predicted_values = [int(value) for value in predicted]
            expected_values = [int(value) for value in expected]
            exact += int(predicted_values == expected_values)
            token_correct += sum(
                left == right
                for left, right in zip(predicted_values, expected_values, strict=True)
            )
            token_count += len(expected_values)
        return EvaluationResult(
            example_count=len(xs),
            loss=None,
            accuracy=None,
            unit="sequence",
            unit_count=len(xs),
            exact_match_accuracy=(exact / len(xs))
            if "exact_match_accuracy" in requested
            else None,
            token_accuracy=(token_correct / token_count)
            if "token_accuracy" in requested
            else None,
        )

    def _iter_batches(self, x: Tensor, t: Tensor):
        size = len(x) - (len(x) % self.batch_size) if self.drop_last else len(x)
        for start in range(0, size, self.batch_size):
            yield x[start : start + self.batch_size], t[start : start + self.batch_size]

    def _num_batches(self, size: int) -> int:
        return (
            size // self.batch_size
            if self.drop_last
            else (size + self.batch_size - 1) // self.batch_size
        )

    def planned_total_updates(self, sample_count: int) -> int:
        epoch_total = self._num_batches(sample_count) * self.max_epochs
        return (
            epoch_total
            if self.max_updates is None
            else min(epoch_total, self.max_updates)
        )

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state["batch_cursor"] = self.batch_cursor
        state["samples_seen"] = self.samples_seen
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        super().load_state_dict(state)
        self.batch_cursor = int(state.get("batch_cursor", 0))
        self.samples_seen = int(state.get("samples_seen", 0))
