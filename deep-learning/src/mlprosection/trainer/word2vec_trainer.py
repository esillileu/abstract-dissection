"""Event-based trainer for full-softmax and negative-sampling Word2Vec."""

from __future__ import annotations

from typing import Literal

from mlprosection import Tensor
from mlprosection.events import EpochEvent, SourceObjectiveSample, UpdateEvent
from mlprosection.trainer.utils import clip_grads

from .internal_objective import InternalObjectiveTrainer


class Word2VecTrainer(InternalObjectiveTrainer):
    def __init__(
        self,
        model,
        optimizer,
        *,
        max_epochs: int,
        batch_size: int,
        max_updates: int | None = None,
        drop_last: bool = True,
        max_grad: float | None = None,
        event_receivers=None,
    ) -> None:
        super().__init__(
            model, optimizer, max_epochs=max_epochs, max_updates=max_updates,
            event_receivers=event_receivers,
        )
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.max_grad = max_grad
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
                xp = self.backend.xp
                order = xp.random.permutation(len(contexts))
                shuffled_contexts, shuffled_targets = contexts[order], targets[order]
                for local_iteration, (batch_x, batch_t) in enumerate(self._iter_batches(shuffled_contexts, shuffled_targets)):
                    source_loss = self.model.forward(batch_x, batch_t)
                    fixed_candidates = self._negative_candidates()
                    self.model.backward()
                    if self.max_grad is not None:
                        clip_grads(list(self.model.named_parameters()), self.max_grad)
                    self.optimizer.update()
                    post_loss = self._post_update_loss(batch_x, batch_t, fixed_candidates)
                    self.global_step += 1
                    sample_count += len(batch_x)
                    self._emit_update(UpdateEvent(
                        update=self.global_step, epoch=self.epoch, batch_size=len(batch_x),
                        loss=post_loss, learning_rate=self._learning_rate(),
                    ))
                    self._emit_source_objective(SourceObjectiveSample(
                        update=self.global_step, epoch=self.epoch,
                        local_iteration=local_iteration, objective=source_loss,
                        unit_count=self._prediction_terms(batch_x, batch_t),
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

    def _post_update_loss(self, x: Tensor, t: Tensor, candidates) -> Tensor:
        if candidates is None:
            return self.model.forward(x, t)
        return self.model.forward(x, t, negative_candidates=candidates)

    def _negative_candidates(self):
        getter = getattr(self.model, "last_negative_candidates", None)
        return getter() if getter is not None else None

    def _prediction_terms(self, x: Tensor, t: Tensor) -> int:
        if getattr(self.model, "architecture", "cbow") == "skipgram":
            return int(t.data.size)
        return len(x)

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
