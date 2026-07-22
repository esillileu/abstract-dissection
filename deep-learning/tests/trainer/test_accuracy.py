import pytest

from mlprosection import Tensor
from mlprosection.nn.layers.base import Layer
from mlprosection.nn.layers.criterion import SoftmaxWithLoss
from mlprosection.trainer.base import Trainer
from mlprosection.trainer import ForwardTrainer


class IdentityModel(Layer):
    def forward_manual(self, x):
        return x

    def backward_manual(self, dout):
        return dout


class DummyTrainer(Trainer):
    def fit(self, *args, **kwargs):
        raise NotImplementedError


class DummyProgress:
    def set_postfix(self, **kwargs):
        pass


def test_run_epoch_records_validation_accuracy_once():
    trainer = DummyTrainer(
        model=IdentityModel(),
        criterion=SoftmaxWithLoss(),
        optimizer=None,
        batch_size=2,
        log_interval=1,
    )
    trainer.train = False
    trainer.pbar = DummyProgress()
    trainer.epoch = 1

    x = Tensor(
        [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.3, 0.7],
        ]
    )
    t = Tensor([1, 1, 0])

    trainer.run_epoch(x, t)

    assert trainer.accuracies.valid == [pytest.approx(1 / 3)]
    assert trainer.accuracies.train == []
    assert [log["eval_step"] for log in trainer.logs.valid] == [1, 2]


class DummyOptimizer:
    def __init__(self) -> None:
        self.params = []

    def update(self) -> None:
        pass


def test_forward_trainer_records_fixed_train_probe_when_enabled():
    trainer = ForwardTrainer(
        model=IdentityModel(),
        criterion=SoftmaxWithLoss(),
        optimizer=DummyOptimizer(),
        max_epoch=1,
        batch_size=2,
        log_interval=None,
        record_first_validation_evaluation=True,
        record_step_validation_interval=2,
        record_step_train_evaluation=True,
    )
    x = Tensor([[0.1, 0.9], [0.8, 0.2], [0.7, 0.3], [0.2, 0.8]])
    t = Tensor([1, 0, 0, 1])

    trainer.fit(x, t, x, t, x[:2], t[:2])

    assert [global_step for _, global_step, _ in trainer.validation_evaluations] == [1, 2]
    for _, _, metrics in trainer.validation_evaluations:
        assert set(metrics) == {"valid/loss", "valid/accuracy", "train/loss", "train/accuracy"}
