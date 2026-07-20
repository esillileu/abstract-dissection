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

        self.epoch: int | None = None
        self.on_epoch_checkpoint = on_epoch_checkpoint

    def fit(
        self,
        x_train: Tensor,
        t_train: Tensor,
        x_val: Tensor | None = None,
        t_val: Tensor | None = None,
    ) -> None:
        xp = self.backend.xp
        self.start_time = time.time()
        self.train = True
        skip_validation = x_val is None or t_val is None

        data_size = len(x_train)
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
                    if self.max_updates is not None and self.global_step >= self.max_updates:
                        break
                    self.epoch = epoch + 1
                    idx = xp.random.permutation(xp.arange(data_size))
                    shuffled_x, shuffled_t = x_train[idx], t_train[idx]
                    self._run_measured_epoch("train", epoch, shuffled_x, shuffled_t)

                    if not skip_validation:
                        self.train = False
                        self._run_measured_epoch("eval", epoch, x_val, t_val)
                        self.train = True
                    if self.on_epoch_checkpoint is not None:
                        self.on_epoch_checkpoint()
                    if self.max_updates is not None and self.global_step >= self.max_updates:
                        break
        finally:
            if self.profiling_config.collect_memory_metrics:
                self.runtime_monitor.snapshot_memory("train.end", synchronize=True)
                self.runtime_monitor.snapshot_memory("run.end")
            self.record_final_memory_metrics()
            self.detail_profiler.stop_run()
            self.train = True

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
