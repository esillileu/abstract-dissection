"""Trainer isolated to the fused negative-sampling Word2Vec stack."""

from __future__ import annotations

from typing import Literal

from mlprosection import Tensor
from mlprosection.events import EpochEvent, SourceObjectiveSample, UpdateEvent

from .word2vec_trainer import Word2VecTrainer


class FusedNegativeSamplingTrainer(Word2VecTrainer):
    """Word2Vec loop that bypasses dense logits and dense gradients."""

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
                shuffled_contexts = contexts[order]
                shuffled_targets = targets[order]
                batches = self._iter_batches(
                    shuffled_contexts,
                    shuffled_targets,
                )
                for local_iteration, (batch_x, batch_t) in enumerate(batches):
                    model_x, objective_t = self.batch_adapter.prepare(
                        batch_x,
                        batch_t,
                    )
                    objective_batch = self.objective.prepare(objective_t)
                    result = self.objective.forward_fused(
                        self.model,
                        model_x,
                        objective_batch,
                        example_count=len(batch_x),
                    )
                    self.objective.backward_fused(self.model)
                    self.optimizer.update()
                    probe_state = self._snapshot_evaluation_state()
                    try:
                        post_result = self.objective.forward_fused(
                            self.model,
                            model_x,
                            objective_batch,
                            cache=False,
                            example_count=len(batch_x),
                        )
                    finally:
                        self._restore_evaluation_state(probe_state)
                    self.global_step += 1
                    sample_count += len(batch_x)
                    self._emit_update(
                        UpdateEvent(
                            update=self.global_step,
                            epoch=self.epoch,
                            batch_size=len(batch_x),
                            loss=(
                                post_result.reporting_loss
                                if post_result.reporting_loss is not None
                                else post_result.loss
                            ),
                            learning_rate=self._learning_rate(),
                            book_loss=post_result.loss,
                        )
                    )
                    self._emit_source_objective(
                        SourceObjectiveSample(
                            update=self.global_step,
                            epoch=self.epoch,
                            local_iteration=local_iteration,
                            objective=(
                                result.reporting_loss
                                if result.reporting_loss is not None
                                else result.loss
                            ),
                            unit_count=result.unit_count,
                            book_objective=result.loss,
                        )
                    )
                    self.batch_cursor = local_iteration + 1
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
