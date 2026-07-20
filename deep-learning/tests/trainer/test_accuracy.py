import pytest

from mlprosection import Tensor
from mlprosection.nn.layers.base import Layer
from mlprosection.nn.layers.criterion import SoftmaxWithLoss
from mlprosection.trainer.base import Trainer


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
