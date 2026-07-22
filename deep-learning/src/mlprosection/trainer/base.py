from __future__ import annotations

import time

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Iterator, TYPE_CHECKING, Iterable
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
from .callbacks import TrainerCallback

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
        criterion: Criterion | None,
        optimizer: Optimizer,
        max_epoch: int = 10,
        max_updates: int | None = None,
        batch_size: int = 32,
        log_interval: int = 20,
        drop_last: bool | None = False,
        profiling_config: ProfilingConfig | None = None,
        callbacks: Iterable[TrainerCallback] | None = None,
    ):
        self.model: Layer = model
        self.criterion: Criterion | None = criterion
        self.optimizer: Optimizer = optimizer

        self.max_epoch = max_epoch
        self.max_updates = max_updates
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
        self.eval_step = 0
        self.callbacks = tuple(callbacks or ())
        self.max_grad: float | None = None
        self.record_step_loss = "none"
        self.step_losses: list[tuple[int, float]] = []
        self.on_train_step: Callable[[int], None] | None = None

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

    def state_dict(self) -> dict[str, object]:
        return {
            "global_step": self.global_step,
            "eval_step": self.eval_step,
            "epoch": getattr(self, "epoch", None),
            "losses": {"train": list(self.losses.train), "valid": list(self.losses.valid)},
            "accuracies": {"train": list(self.accuracies.train), "valid": list(self.accuracies.valid)},
            "logs": {"train": list(self.logs.train), "valid": list(self.logs.valid)},
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.global_step = int(state["global_step"])
        self.eval_step = int(state.get("eval_step", 0))
        if hasattr(self, "epoch"):
            self.epoch = state.get("epoch")
        losses = state.get("losses", {})
        accuracies = state.get("accuracies", {})
        logs = state.get("logs", {})
        if isinstance(losses, dict):
            self.losses = TVListContainer(list(losses.get("train", [])), list(losses.get("valid", [])))
        if isinstance(accuracies, dict):
            self.accuracies = TVListContainer(list(accuracies.get("train", [])), list(accuracies.get("valid", [])))
        if isinstance(logs, dict):
            self.logs = TVListContainer(list(logs.get("train", [])), list(logs.get("valid", [])))

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
        if self.criterion is None:
            raise RuntimeError("this trainer requires a specialized step implementation")
        self.model.train(self.train)

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
                        self.clip_gradients()

                with self.detail_profiler.section(
                    "optimizer_update",
                    enabled=profile,
                ):
                    self.optimizer.update()

        return loss, pred

    def clip_gradients(self) -> None:
        if self.max_grad is not None:
            clip_grads(list(self.model.named_parameters()), self.max_grad)

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
            if self.train and self.record_step_loss != "none":
                recorded_loss = loss
                if self.record_step_loss == "post_update":
                    if self.criterion is None:
                        raise RuntimeError("step-loss recording requires a criterion")
                    recorded_loss = self.criterion.forward(self.model.forward(batch_x), batch_t)
                self.step_losses.append((self.global_step + 1, float(recorded_loss.data)))

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
                self._emit_batch_end()
                if self.on_train_step is not None:
                    self.on_train_step(self.global_step)

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

            if self.train and self.max_updates is not None and self.global_step >= self.max_updates:
                break

        if sample_count > 0:
            self.interval_log(
                iters=iters,
                total_loss=total_loss,
                sample_count=sample_count,
            )

        accuracy = self.compute_accuracy(correct_count, epoch_sample_count)
        self.record_accuracy(accuracy)
        self._emit_epoch_end(epoch=self.epoch or 0, accuracy=accuracy)

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

    def start_profiling_run(self) -> None:
        """Start the common telemetry lifecycle used by non-forward trainers."""
        self.detail_profiler.start_run()
        if self.profiling_config.collect_memory_metrics:
            self.runtime_monitor.snapshot_memory("run.start", synchronize=True)
            self.runtime_monitor.snapshot_memory("train.start")
        self.record_model_metrics()

    def finish_profiling_run(self) -> None:
        """Persist final common telemetry after a specialized training loop."""
        if self.profiling_config.collect_memory_metrics:
            self.runtime_monitor.snapshot_memory("train.end", synchronize=True)
            self.runtime_monitor.snapshot_memory("run.end")
        self.record_final_memory_metrics()
        self.detail_profiler.stop_run()

    def begin_profiled_epoch(self, *, split: str, epoch_index: int) -> int | None:
        if self.profiling_config.collect_memory_metrics:
            self.runtime_monitor.snapshot_memory(
                f"epoch.{epoch_index}.{split}.start", synchronize=True
            )
        if not self.profiling_config.collect_epoch_metrics:
            return None
        return time.perf_counter_ns()

    def finish_profiled_epoch(
        self,
        *,
        split: str,
        epoch_index: int,
        sample_count: int,
        started_ns: int | None,
    ) -> None:
        if started_ns is not None:
            self.backend_profiler.synchronize()
            duration_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
            self.runtime_monitor.set_metric(
                f"runtime.epoch.{epoch_index}.{split}_duration_ms", duration_ms
            )
            self.runtime_monitor.set_metric(
                f"throughput.epoch.{epoch_index}.{split}_samples_per_s",
                0.0 if duration_ms <= 0 else sample_count / (duration_ms / 1_000),
            )
        if self.profiling_config.collect_memory_metrics:
            self.runtime_monitor.snapshot_memory(
                # The elapsed-time branch already synchronized the current stream.
                f"epoch.{epoch_index}.{split}.end", synchronize=started_ns is None
            )

    def internal_loss_step(self, x: Tensor, t: Tensor) -> Tensor:
        """Run a model-owned loss update with the standard detailed telemetry."""
        profile = self.profiling_controller.should_profile(self.global_step)
        if self.profiling_controller.should_sample_memory(self.global_step):
            self.runtime_monitor.update_memory_peaks()
        if profile and self.profiling_config.profile_memory:
            self.runtime_monitor.snapshot_memory(
                f"profile.step.{self.global_step}.before", synchronize=True
            )
        with self.detail_profiler.section("train_step", enabled=profile):
            with self.detail_profiler.section("forward", enabled=profile):
                loss = self.model.forward(x, t)
            with self.detail_profiler.section("backward", enabled=profile):
                self.model.backward()
            if self.max_grad is not None:
                with self.detail_profiler.section("gradient_clip", enabled=profile):
                    self.clip_gradients()
            with self.detail_profiler.section("optimizer_update", enabled=profile):
                self.optimizer.update()
        if profile and self.profiling_config.profile_memory:
            self.runtime_monitor.snapshot_memory(
                f"profile.step.{self.global_step}.after", synchronize=True
            )
        return loss

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

        log = {
            "epoch": self.epoch,
            "iteration": iters + 1,
            "loss": loss_value,
            "elapsed_time": elapsed,
        }

        if self.train:
            # This is the completed update count, not the per-epoch iteration.
            # It remains monotonic across epochs and checkpoint resume.
            log["global_step"] = self.global_step
            self.losses.train.append(loss_value)
            self.logs.train.append(log)
        else:
            self.eval_step += 1
            log["eval_step"] = self.eval_step
            self.losses.valid.append(loss_value)
            self.logs.valid.append(log)
        self._emit_interval(log)

    def _emit_batch_end(self) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(step=self.global_step)

    def _emit_interval(self, log: dict[str, float | int]) -> None:
        metrics = {key: float(value) for key, value in log.items() if isinstance(value, (int, float))}
        for callback in self.callbacks:
            callback.on_interval(metrics=metrics)

    def _emit_epoch_end(self, *, epoch: int, accuracy: float) -> None:
        split = "train" if self.train else "valid"
        self.emit_epoch_metrics(epoch=epoch, metrics={f"{split}/accuracy": accuracy})

    def emit_epoch_metrics(self, *, epoch: int, metrics: dict[str, float]) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(epoch=epoch, metrics=metrics)
