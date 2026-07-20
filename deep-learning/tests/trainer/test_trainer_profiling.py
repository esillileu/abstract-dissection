from mlprosection import Tensor
from mlprosection.nn.layers.base import Layer
from mlprosection.nn.layers.criterion import SoftmaxWithLoss
from mlprosection.profiling import ProfilingConfig
from mlprosection.trainer import ForwardTrainer


class IdentityModel(Layer):
    def forward_manual(self, x):
        return x

    def backward_manual(self, dout):
        return dout


class DummyOptimizer:
    def __init__(self) -> None:
        self.params = []
        self.update_count = 0

    def update(self) -> None:
        self.update_count += 1


class RecordingCallback:
    def __init__(self) -> None:
        self.batch_steps: list[int] = []
        self.intervals: list[dict[str, float]] = []
        self.epochs: list[tuple[int, dict[str, float]]] = []

    def on_batch_end(self, *, step: int) -> None:
        self.batch_steps.append(step)

    def on_interval(self, *, metrics: dict[str, float]) -> None:
        self.intervals.append(metrics)

    def on_epoch_end(self, *, epoch: int, metrics: dict[str, float]) -> None:
        self.epochs.append((epoch, metrics))


def test_forward_trainer_collects_common_profiling_metrics() -> None:
    trainer = ForwardTrainer(
        model=IdentityModel(),
        criterion=SoftmaxWithLoss(),
        optimizer=DummyOptimizer(),
        max_epoch=2,
        batch_size=2,
        log_interval=None,
    )

    x = Tensor([[0.1, 0.9], [0.8, 0.2], [0.7, 0.3], [0.2, 0.8]])
    t = Tensor([1, 0, 0, 1])

    trainer.fit(x, t)
    metrics = trainer.profiling_metrics()

    assert trainer.global_step == 4
    assert "runtime.train_total.mean_ms" in metrics
    assert "runtime.epoch.train.count" in metrics
    assert "runtime.epoch.0.train_duration_ms" in metrics
    assert "throughput.epoch.0.train_samples_per_s" in metrics
    assert "memory.run.start.cpu.rss_bytes" in metrics
    assert metrics["profiling.enabled"] == 0
    assert "runtime.profile.forward.count" not in metrics


def test_forward_trainer_profiles_only_configured_steps() -> None:
    trainer = ForwardTrainer(
        model=IdentityModel(),
        criterion=SoftmaxWithLoss(),
        optimizer=DummyOptimizer(),
        max_epoch=2,
        batch_size=2,
        log_interval=None,
        profiling_config=ProfilingConfig(
            enabled=True,
            start_step=1,
            num_steps=2,
            profile_memory=True,
        ),
    )

    x = Tensor([[0.1, 0.9], [0.8, 0.2], [0.7, 0.3], [0.2, 0.8]])
    t = Tensor([1, 0, 0, 1])

    trainer.fit(x, t)
    metrics = trainer.profiling_metrics()

    assert trainer.global_step == 4
    assert metrics["runtime.profile.train_step.count"] == 2
    assert metrics["runtime.profile.forward.count"] == 2
    assert metrics["runtime.profile.backward.count"] == 2
    assert metrics["runtime.profile.optimizer_update.count"] == 2
    assert "memory.profile.step.1.before.cpu.rss_bytes" in metrics
    assert "memory.profile.step.2.after.cpu.rss_bytes" in metrics
    assert "memory.profile.step.0.before.cpu.rss_bytes" not in metrics


def test_forward_trainer_emits_plain_callback_events() -> None:
    callback = RecordingCallback()
    trainer = ForwardTrainer(
        model=IdentityModel(), criterion=SoftmaxWithLoss(), optimizer=DummyOptimizer(),
        max_epoch=1, batch_size=2, log_interval=1, callbacks=[callback],
    )
    x = Tensor([[0.1, 0.9], [0.8, 0.2], [0.7, 0.3], [0.2, 0.8]])
    t = Tensor([1, 0, 0, 1])

    trainer.fit(x, t)

    assert callback.batch_steps == [1, 2]
    assert callback.intervals and all("loss" in event for event in callback.intervals)
    assert [log["global_step"] for log in trainer.logs.train] == [1, 2]
    assert trainer.eval_step == 0
    assert callback.epochs == [(1, {"train/accuracy": 1.0})]
