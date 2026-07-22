from __future__ import annotations

import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Iterable, Callable

from mlprosection.profiling import ProfilingConfig
from .base import Trainer
from .callbacks import TrainerCallback

if TYPE_CHECKING:
    from mlprosection import Tensor
    from mlprosection.nn.types import Layer, Criterion
    from mlprosection.optim import Optimizer


class ForwardTrainer(Trainer):
    def __init__(
        self,
        model: Layer,
        criterion: Criterion,
        optimizer: Optimizer,
        max_epoch: int = 10,
        max_updates: int | None = None,
        batch_size: int = 32,
        log_interval: int = 20,
        max_grad: float | None = None,
        drop_last: bool | None = False,
        sampling_method: str = "permutation_per_epoch",
        record_step_loss: str = "none",
        record_first_step_evaluation: bool = False,
        record_epoch_evaluation: bool = False,
        record_step_evaluation_interval: int | None = None,
        record_first_validation_evaluation: bool = False,
        record_step_validation_interval: int | None = None,
        record_step_train_evaluation: bool = False,
        profiling_config: ProfilingConfig | None = None,
        callbacks: Iterable[TrainerCallback] | None = None,
        on_epoch_checkpoint: Callable[[], None] | None = None,
    ):
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            max_epoch=max_epoch,
            max_updates=max_updates,
            batch_size=batch_size,
            log_interval=log_interval,
            drop_last=drop_last,
            profiling_config=profiling_config,
            callbacks=callbacks,
        )
        self.max_grad = max_grad
        if sampling_method not in {"permutation_per_epoch", "with_replacement"}:
            raise ValueError(f"unsupported sampling_method: {sampling_method}")
        if record_step_loss not in {"none", "pre_update", "post_update"}:
            raise ValueError(f"unsupported record_step_loss mode: {record_step_loss}")
        self.sampling_method = sampling_method
        self.record_step_loss = record_step_loss
        self.step_losses: list[tuple[int, float]] = []
        self.record_first_step_evaluation = record_first_step_evaluation
        self.record_epoch_evaluation = record_epoch_evaluation
        if (
            record_step_evaluation_interval is not None
            and record_step_evaluation_interval < 1
        ):
            raise ValueError("record_step_evaluation_interval must be positive")
        self.record_step_evaluation_interval = record_step_evaluation_interval
        self.graph_evaluations: list[tuple[int, dict[str, float]]] = []
        if record_step_validation_interval is not None and record_step_validation_interval < 1:
            raise ValueError("record_step_validation_interval must be positive")
        self.record_first_validation_evaluation = record_first_validation_evaluation
        self.record_step_validation_interval = record_step_validation_interval
        self.record_step_train_evaluation = record_step_train_evaluation
        self.validation_evaluations: list[tuple[int, int, dict[str, float]]] = []

        self.epoch: int | None = None
        self.on_epoch_checkpoint = on_epoch_checkpoint

    def fit(
        self,
        x_train: Tensor,
        t_train: Tensor,
        x_val: Tensor | None = None,
        t_val: Tensor | None = None,
        x_train_probe: Tensor | None = None,
        t_train_probe: Tensor | None = None,
    ) -> None:
        xp = self.backend.xp
        self.start_time = time.time()
        self.train = True
        skip_validation = x_val is None or t_val is None

        data_size = len(x_train)

        def record_first_step(step: int) -> None:
            if step == 1:
                self.graph_evaluations.append((
                    0,
                    self._full_evaluation(x_train, t_train, x_val, t_val),
                ))
            elif (
                self.record_step_evaluation_interval
                and (step - 1) % self.record_step_evaluation_interval == 0
            ):
                self.graph_evaluations.append((
                    step // self.record_step_evaluation_interval,
                    self._full_evaluation(x_train, t_train, x_val, t_val),
                ))

        def record_interval_evaluation(step: int) -> None:
            if skip_validation and not self.record_step_train_evaluation:
                return
            should_record = (
                self.record_first_validation_evaluation and step == 1
            ) or (
                self.record_step_validation_interval is not None
                and step % self.record_step_validation_interval == 0
            )
            if should_record:
                values = {}
                if not skip_validation:
                    values.update({
                        f"valid/{key}": value
                        for key, value in self._evaluate_split(x_val, t_val).items()
                    })
                if self.record_step_train_evaluation:
                    if x_train_probe is None or t_train_probe is None:
                        raise ValueError("step train evaluation requires a train probe")
                    values.update({
                        f"train/{key}": value
                        for key, value in self._evaluate_split(x_train_probe, t_train_probe).items()
                    })
                self.validation_evaluations.append((
                    len(self.validation_evaluations), step,
                    values,
                ))

        step_recorders = []
        if self.record_first_step_evaluation:
            step_recorders.append(record_first_step)
        if self.record_first_validation_evaluation or self.record_step_validation_interval:
            step_recorders.append(record_interval_evaluation)
        self.on_train_step = (
            (lambda step: [recorder(step) for recorder in step_recorders])
            if step_recorders else None
        )
        self.detail_profiler.start_run()
        try:
            if self.profiling_config.collect_memory_metrics:
                self.runtime_monitor.snapshot_memory("run.start", synchronize=True)
                self.runtime_monitor.snapshot_memory("train.start")
            self.record_model_metrics()

            if self.profiling_config.collect_common_metrics:
                train_timer = self.runtime_monitor.timer(
                    "train_total",
                    synchronize=True,
                )
            else:
                train_timer = nullcontext()

            with train_timer:
                start_epoch = int(self.epoch or 0)
                for epoch in range(start_epoch, self.max_epoch):
                    if (
                        self.max_updates is not None
                        and self.global_step >= self.max_updates
                    ):
                        break
                    self.epoch = epoch + 1
                    if self.sampling_method == "with_replacement":
                        updates_per_epoch = data_size // self.batch_size
                        if updates_per_epoch == 0:
                            raise ValueError(
                                "dataset is smaller than one training batch"
                            )
                        idx = xp.random.randint(
                            0, data_size, size=updates_per_epoch * self.batch_size
                        )
                    else:
                        idx = xp.random.permutation(xp.arange(data_size))
                    shuffled_x, shuffled_t = x_train[idx], t_train[idx]
                    self._run_measured_epoch("train", epoch, shuffled_x, shuffled_t)

                    if self.record_epoch_evaluation:
                        self.graph_evaluations.append((
                            epoch + 1,
                            self._full_evaluation(x_train, t_train, x_val, t_val),
                        ))

                    if not skip_validation:
                        self.train = False
                        self._run_measured_epoch("eval", epoch, x_val, t_val)
                        self.train = True
                    if self.on_epoch_checkpoint is not None:
                        self.on_epoch_checkpoint()
                    if (
                        self.max_updates is not None
                        and self.global_step >= self.max_updates
                    ):
                        break
        finally:
            self.on_train_step = None
            if self.profiling_config.collect_memory_metrics:
                self.runtime_monitor.snapshot_memory("train.end", synchronize=True)
                self.runtime_monitor.snapshot_memory("run.end")
            self.record_final_memory_metrics()
            self.detail_profiler.stop_run()
            self.train = True

    def _full_evaluation(
        self,
        x_train: Tensor,
        t_train: Tensor,
        x_test: Tensor | None,
        t_test: Tensor | None,
    ) -> dict[str, float]:
        values = self._evaluate_split(x_train, t_train)
        if x_test is not None and t_test is not None:
            values.update({
                f"test/{key}": value
                for key, value in self._evaluate_split(x_test, t_test).items()
            })
        return {
            f"train/{key}": value
            for key, value in values.items()
            if not key.startswith("test/")
        } | {key: value for key, value in values.items() if key.startswith("test/")}

    def _evaluate_split(self, x: Tensor, t: Tensor) -> dict[str, float]:
        was_training = self.train
        self.model.train(False)
        total_loss = 0.0
        samples = 0
        correct = 0
        for batch_x, batch_t in self.iter_batches(x, t):
            y = self.model.forward(batch_x)
            loss = self.criterion.forward(y, batch_t)
            batch_size = len(batch_x)
            total_loss += float(loss.data) * batch_size
            samples += batch_size
            correct += self.count_correct(y, batch_t)
        self.model.train(was_training)
        return {"loss": total_loss / samples, "accuracy": correct / samples}

    def _run_measured_epoch(
        self,
        split: str,
        epoch_index: int,
        x: Tensor,
        t: Tensor,
    ) -> None:
        sample_count = len(x)
        if self.profiling_config.collect_memory_metrics:
            self.runtime_monitor.snapshot_memory(
                f"epoch.{epoch_index}.{split}.start",
                synchronize=True,
            )

        if self.profiling_config.collect_epoch_metrics:
            start = time.perf_counter_ns()
            epoch_timer = self.runtime_monitor.timer(
                f"epoch.{split}",
                synchronize=True,
            )
        else:
            start = None
            epoch_timer = nullcontext()

        with epoch_timer:
            self.run_epoch(x, t)

        if self.profiling_config.collect_epoch_metrics:
            self.backend_profiler.synchronize()
            duration_ms = (time.perf_counter_ns() - start) / 1_000_000
            self.runtime_monitor.set_metric(
                f"runtime.epoch.{epoch_index}.{split}_duration_ms",
                duration_ms,
            )
            throughput = 0.0
            if duration_ms > 0:
                throughput = sample_count / (duration_ms / 1_000)
            self.runtime_monitor.set_metric(
                f"throughput.epoch.{epoch_index}.{split}_samples_per_s",
                throughput,
            )

        if self.profiling_config.collect_memory_metrics:
            self.runtime_monitor.snapshot_memory(
                f"epoch.{epoch_index}.{split}.end",
                synchronize=True,
            )
