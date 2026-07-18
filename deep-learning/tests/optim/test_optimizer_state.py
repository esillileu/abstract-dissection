from __future__ import annotations

import numpy as np

from mlprosection import Tensor
from mlprosection.nn.types import Parameter
from mlprosection.optim.SGD import AdaGrad, Adam, Momentum, RMSprop


def _param() -> Parameter:
    param = Parameter(Tensor([1.0, -1.0], backend="cpu"))
    param.grad[...] = np.array([0.5, -0.25])
    return param


def test_momentum_initializes_state_from_params() -> None:
    param = _param()
    optimizer = Momentum([("w", param)], lr=0.1, momentum=0.9)

    optimizer.update()

    assert "w" in optimizer.v
    np.testing.assert_allclose(optimizer.v["w"], np.array([-0.05, 0.025]))


def test_adagrad_persists_accumulator() -> None:
    param = _param()
    optimizer = AdaGrad([("w", param)], lr=0.1)

    optimizer.update()

    np.testing.assert_allclose(optimizer.h["w"], np.array([0.25, 0.0625]))


def test_rmsprop_persists_accumulator() -> None:
    param = _param()
    optimizer = RMSprop([("w", param)], lr=0.1, decay_rate=0.9)

    optimizer.update()

    np.testing.assert_allclose(optimizer.h["w"], np.array([0.025, 0.00625]))


def test_adam_advances_bias_correction_step() -> None:
    param = _param()
    optimizer = Adam([("w", param)], lr=0.1)

    optimizer.update()

    assert optimizer.iter == 1
    assert optimizer.lr_t != optimizer.lr
    assert np.any(optimizer.m["w"])
    assert np.any(optimizer.v["w"])
