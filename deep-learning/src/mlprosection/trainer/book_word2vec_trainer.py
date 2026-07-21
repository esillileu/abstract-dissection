"""Book-compatible update logging for the word2vec chapter experiments."""

from __future__ import annotations

import time

from mlprosection import Tensor
from mlprosection.nn.layers import Layer
from mlprosection.optim import Optimizer

from .internal_loss_trainer import InternalLossHistory, InternalLossTrainer


class BookWord2VecTrainer(InternalLossTrainer):
    """Reproduce the ch04 ``Trainer.fit`` update and loss-recording semantics.

    Sampling and model kernels remain portable, but batches are shuffled once per
    epoch, partial batches are discarded, and loss is recorded after update 1 and
    then every ``log_interval`` zero-based iteration, as in the book source.
    """

    def __init__(
        self,
        model: Layer,
        optimizer: Optimizer,
        *,
        max_epoch: int,
        batch_size: int,
        prediction_term_count: int,
        max_updates: int | None = None,
        log_interval: int = 20,
        max_grad: float | None = None,
        callbacks=None,
    ) -> None:
        if prediction_term_count < 1:
            raise ValueError("prediction_term_count must be at least 1")
        super().__init__(
            model,
            optimizer,
            max_epoch=max_epoch,
            max_updates=max_updates,
            batch_size=batch_size,
            log_interval=log_interval,
            max_grad=max_grad,
            drop_last=True,
            callbacks=callbacks,
        )
        self.prediction_term_count = prediction_term_count

    def fit(self, xs: Tensor, ts: Tensor) -> InternalLossHistory:
        if len(xs) != len(ts):
            raise ValueError("inputs and targets must have the same sample count")
        if self.num_batches(len(xs)) == 0:
            raise ValueError("dataset is smaller than one training batch")

        self.start_time = time.time()
        self.train = True
        interval_total = 0.0
        interval_count = 0
        for epoch_index in range(self.epoch, self.max_epoch):
            if self.max_updates is not None and self.global_step >= self.max_updates:
                break
            self.epoch = epoch_index + 1
            self.model.train(True)
            order = self.backend.xp.random.permutation(len(xs))
            shuffled_x, shuffled_t = xs[order], ts[order]
            epoch_total = 0.0
            epoch_count = 0

            for iteration, (batch_x, batch_t) in enumerate(self.iter_batches(shuffled_x, shuffled_t)):
                loss = self.model.forward(batch_x, batch_t)
                self.model.backward()
                self.clip_gradients()
                self.optimizer.update()

                self.global_step += 1
                value = float(loss.data)
                epoch_total += value
                epoch_count += 1
                interval_total += value
                interval_count += 1
                self._emit_batch_end()

                if iteration % self.log_interval == 0:
                    average = interval_total / interval_count
                    self.history.interval_loss.append(average)
                    self.losses.train.append(average)
                    log = {
                        "epoch": self.epoch,
                        "iteration": iteration + 1,
                        "global_step": self.global_step,
                        "loss": average,
                        "normalized_loss": average / self.prediction_term_count,
                        "elapsed_time": time.time() - self.start_time,
                    }
                    self.logs.train.append(log)
                    self._emit_interval(log)
                    interval_total = 0.0
                    interval_count = 0

                if self.max_updates is not None and self.global_step >= self.max_updates:
                    break

            epoch_loss = epoch_total / epoch_count
            self.history.epoch_loss.append(epoch_loss)
            self.emit_epoch_metrics(
                epoch=self.epoch,
                metrics={
                    "train/loss": epoch_loss,
                    "train/normalized_loss": epoch_loss / self.prediction_term_count,
                },
            )
            if self.max_updates is not None and self.global_step >= self.max_updates:
                break
        return self.history
