from __future__ import annotations

import time

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Iterator, TYPE_CHECKING
from dataclasses import dataclass, field
from mlprosection.profiling import ProfilingConfig
from mlprosection.profiling.backend import create_backend_profiler
from mlprosection.profiling.controller import ProfilingController
from mlprosection.profiling.detail import DetailProfiler
from mlprosection.profiling.monitor import RuntimeMonitor
from mlprosection.profiling.utils import (
    count_gradient_bytes,
    count_optimizer_state_bytes,
    count_parameter_bytes,
    count_parameter_elements,
)
from .utils import clip_grads

if TYPE_CHECKING:
    from mlprosection import Tensor
    from mlprosection.optim import Optimizer
    from mlprosection.nn.types import Layer, Criterion


@dataclass
class TVListContainer:
    train: List = field(default_factory=list)
    valid: List = field(default_factory=list)


class Trainer(ABC):
    def __init__(
        self,
        model: Layer,
        criterion: Criterion,
        optimizer: Optimizer,
        max_epoch: int = 10,
        batch_size: int = 32,
        log_interval: int = 20,
        drop_last: bool | None = False,
        profiling_config: ProfilingConfig | None = None,
    ):
        self.model: Layer = model
        self.criterion: Criterion = criterion
        self.optimizer: Optimizer = optimizer

        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.log_interval: int = log_interval
        self.drop_last = drop_last

        self.losses: TVListContainer = TVListContainer()
        self.accuracies: TVListContainer = TVListContainer()
        self.logs: TVListContainer = TVListContainer()

        self.start_time: float = 0.0
        self.train: bool = True
        self.profiling_config = profiling_config or ProfilingConfig()
        self.backend_profiler = create_backend_profiler(self.backend)
        self.runtime_monitor = RuntimeMonitor(self.backend_profiler)
        self.profiling_controller = ProfilingController(self.profiling_config)
        self.detail_profiler = DetailProfiler(
            self.profiling_config,
            self.backend_profiler,
            self.runtime_monitor,
        )
        self.global_step = 0

    @property
    def backend(self):
        return self.model.backend

    @property
    def dtype(self):
        return self.model.dtype

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def plot(self, *args, **kwargs):
        raise NotImplementedError

    def num_batches(self, data_size: int) -> int:
        if self.drop_last:
            return data_size // self.batch_size

        return (data_size + self.batch_size - 1) // self.batch_size

    def iter_batches(self, x, t)-> Iterator[tuple[Tensor, Tensor]]:
        size = len(x)

        if self.drop_last:
            size -= size % self.batch_size

        for start in range(0, size, self.batch_size):
            end = start + self.batch_size
            yield x[start:end], t[start:end]

    def step(
        self,
        x: Tensor,
        t: Tensor,
        *,
        profile: bool = False,
    ) -> tuple[Tensor, Tensor]:
        with self.detail_profiler.section("train_step", enabled=profile):
            with self.detail_profiler.section("forward", enabled=profile):
                y = self.model.forward(x)
                loss = self.criterion.forward(y, t)
            pred = self.criterion.y if self.criterion.y is not None else y

            if self.train:
                with self.detail_profiler.section("backward", enabled=profile):
                    dx = self.criterion.backward()
                    self.model.backward(dx)

                if self.max_grad is not None:
                    with self.detail_profiler.section(
                        "gradient_clip",
                        enabled=profile,
                    ):
                        clip_grads(
                            list(self.model.named_parameters()),
                            self.max_grad,
                        )

                with self.detail_profiler.section(
                    "optimizer_update",
                    enabled=profile,
                ):
                    self.optimizer.update()

        return loss, pred

    def count_correct(self, y: Tensor, t: Tensor) -> int:
        y_data = y.data
        t_data = t.data

        if t_data.size == y_data.size and t_data.ndim > 1:
            labels = t_data.argmax(axis=1)
        else:
            labels = t_data.reshape(-1)

        if y_data.ndim > 1 and y_data.shape[1] > 1:
            predictions = y_data.argmax(axis=1)
        else:
            predictions = (y_data.reshape(-1) >= 0.5).astype(labels.dtype)

        correct = (predictions == labels).sum()
        return self.backend.scalar_to_int(correct)

    def compute_accuracy(self, correct_count: int, sample_count: int) -> float:
        if sample_count == 0:
            return 0.0

        return correct_count / sample_count

    def record_accuracy(self, accuracy: float) -> None:
        if self.train:
            self.accuracies.train.append(accuracy)
        else:
            self.accuracies.valid.append(accuracy)

    def run_epoch(self, x:Tensor, t:Tensor) -> None:
        xp = self.backend.xp

        total_loss = xp.asarray(0.0, dtype=x.dtype)
        sample_count = 0
        epoch_sample_count = 0
        correct_count = 0
        batch_count = 0

        for iters, (batch_x, batch_t) in enumerate(self.iter_batches(x, t)):
            profile_this_step = (
                self.train
                and self.profiling_controller.should_profile(self.global_step)
            )
            if (
                self.train
                and self.profiling_controller.should_sample_memory(self.global_step)
            ):
                self.runtime_monitor.update_memory_peaks()

            if profile_this_step and self.profiling_config.profile_memory:
                self.runtime_monitor.snapshot_memory(
                    f"profile.step.{self.global_step}.before",
                    synchronize=True,
                )

            current_size = batch_x.shape[0]
            loss, y = self.step(batch_x, batch_t, profile=profile_this_step)

            if profile_this_step and self.profiling_config.profile_memory:
                self.runtime_monitor.snapshot_memory(
                    f"profile.step.{self.global_step}.after",
                    synchronize=True,
                )

            total_loss += loss.data * current_size
            sample_count += current_size
            epoch_sample_count += current_size
            correct_count += self.count_correct(y, batch_t)
            batch_count += 1
            if self.train:
                self.global_step += 1

            if self.train:
                self.pbar.update(1)

            should_log = (
                self.log_interval is not None and batch_count >= self.log_interval
            )

            if should_log:
                self.interval_log(
                    iters=iters,
                    total_loss=total_loss,
                    sample_count=sample_count,
                )

                total_loss = xp.asarray(0.0, dtype=x.dtype)
                sample_count = 0
                batch_count = 0

        if sample_count > 0:
            self.interval_log(
                iters=iters,
                total_loss=total_loss,
                sample_count=sample_count,
            )

        accuracy = self.compute_accuracy(correct_count, epoch_sample_count)
        self.record_accuracy(accuracy)

    def record_model_metrics(self) -> None:
        if not self.profiling_config.collect_model_metrics:
            return

        self.runtime_monitor.set_metric(
            "model.parameter_count",
            count_parameter_elements(self.model),
        )
        self.runtime_monitor.set_metric(
            "model.parameter_bytes",
            count_parameter_bytes(self.model),
        )
        self.runtime_monitor.set_metric(
            "model.gradient_bytes",
            count_gradient_bytes(self.model),
        )
        self.runtime_monitor.set_metric(
            "optimizer.state_bytes",
            count_optimizer_state_bytes(self.optimizer),
        )
        self.runtime_monitor.set_metric(
            "profiling.enabled",
            int(self.profiling_config.enabled),
        )
        self.runtime_monitor.set_metric(
            "profiling.profiled_step_count",
            self.profiling_config.num_steps if self.profiling_config.enabled else 0,
        )

    def record_final_memory_metrics(self) -> None:
        self.runtime_monitor.set_metric(
            "model.gradient_bytes",
            count_gradient_bytes(self.model),
        )
        self.runtime_monitor.set_metric(
            "optimizer.state_bytes",
            count_optimizer_state_bytes(self.optimizer),
        )

    def profiling_metrics(self) -> dict[str, int | float]:
        return self.runtime_monitor.metrics()

    def dump_profiling_artifacts(self, output_dir: str | Path) -> list[Path]:
        return self.detail_profiler.dump_artifacts(output_dir)

    def interval_log(
        self,
        iters: int,
        total_loss,
        sample_count: int,
    ) -> None:
        avg_loss = total_loss / sample_count
        loss_value = float(avg_loss.item())

        elapsed = time.time() - self.start_time

        self.pbar.set_postfix(
            epoch=f"{self.epoch}/{self.max_epoch}",
            loss=f"{loss_value:.4f}",
            elapsed=f"{elapsed:.1f}s",
        )

        log = {
            "epoch": self.epoch,
            "iteration": iters + 1,
            "loss": loss_value,
            "elapsed_time": elapsed,
        }

        if self.train:
            self.losses.train.append(loss_value)
            self.logs.train.append(log)
        else:
            self.losses.valid.append(loss_value)
            self.logs.valid.append(log)
