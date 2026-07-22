from __future__ import annotations

import pytest

from mlprosection import Tensor
from mlprosection.nn.layers.base import Layer
from mlprosection.nn.layers.criterion import SoftmaxWithLoss
from mlprosection.trainer import ForwardTrainer


class IdentityModel(Layer):
    def forward_manual(self, x):
        return x

    def backward_manual(self, dout):
        return dout


class DummyOptimizer:
    def __init__(self) -> None:
        self.params = []
        self.lr = 0.1

    def update(self) -> None:
        pass


def test_evaluate_returns_requested_metrics_and_restores_training_mode() -> None:
    trainer = ForwardTrainer(
        IdentityModel(), SoftmaxWithLoss(), DummyOptimizer(), max_epochs=1, batch_size=2
    )
    x = Tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    t = Tensor([1, 0, 0])

    trainer.model.train(True)
    result = trainer.evaluate(x, t)

    assert result.example_count == 3
    assert result.loss is not None
    assert result.accuracy == pytest.approx(2 / 3)
    assert trainer.model.training is True


def test_evaluate_can_skip_loss_or_accuracy() -> None:
    trainer = ForwardTrainer(
        IdentityModel(), SoftmaxWithLoss(), DummyOptimizer(), max_epochs=1, batch_size=2
    )
    x = Tensor([[0.1, 0.9], [0.8, 0.2]])
    t = Tensor([1, 0])

    result = trainer.evaluate(x, t, metrics=("accuracy",))

    assert result.loss is None
    assert result.accuracy == pytest.approx(1.0)
