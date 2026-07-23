from __future__ import annotations

import numpy as np

from mlprosection import Tensor
from mlprosection.nn.layers import (
    TimeAffine,
    TimeBiLSTM,
    TimeEmbedding,
    TimeLSTM,
    TimeRNN,
)
from mlprosection.nn.objective import TemporalSoftmaxCrossEntropy


def test_time_affine_forward_backward_shapes() -> None:
    layer = TimeAffine(3, 5, backend="cpu")
    x = Tensor(np.random.randn(2, 4, 3).astype("f"), backend="cpu")

    out = layer.forward(x)
    dx = layer.backward(Tensor(np.ones_like(out.data), backend="cpu"))

    assert out.shape == (2, 4, 5)
    assert dx.shape == x.shape
    assert layer.W.grad.shape == layer.W.shape
    assert layer.b.grad.shape == layer.b.shape


def test_time_embedding_accumulates_repeated_indices() -> None:
    layer = TimeEmbedding(4, 3, backend="cpu")
    xs = Tensor(np.array([[0, 1, 0], [2, 1, 3]]), backend="cpu")

    out = layer.forward(xs)
    layer.backward(Tensor(np.ones_like(out.data), backend="cpu"))

    assert out.shape == (2, 3, 3)
    np.testing.assert_array_equal(layer.W.grad[0], np.full(3, 2.0))
    np.testing.assert_array_equal(layer.W.grad[1], np.full(3, 2.0))


def test_time_lstm_stateful_reset() -> None:
    layer = TimeLSTM(3, 4, stateful=True, backend="cpu")
    xs = Tensor(np.random.randn(2, 5, 3).astype("f"), backend="cpu")

    out = layer.forward(xs)
    dx = layer.backward(Tensor(np.ones_like(out.data), backend="cpu"))

    assert out.shape == (2, 5, 4)
    assert dx.shape == xs.shape
    assert layer.h is not None

    layer.reset_state()

    assert layer.h is None
    assert layer.c is None


def test_time_rnn_and_bilstm_shapes() -> None:
    xs = Tensor(np.random.randn(2, 5, 3).astype("f"), backend="cpu")
    rnn = TimeRNN(3, 4, backend="cpu")
    bilstm = TimeBiLSTM(3, 4, backend="cpu")

    rnn_out = rnn.forward(xs)
    bilstm_out = bilstm.forward(xs)

    assert rnn_out.shape == (2, 5, 4)
    assert bilstm_out.shape == (2, 5, 8)


def test_time_softmax_with_loss_ignores_label() -> None:
    objective = TemporalSoftmaxCrossEntropy()
    scores = Tensor(np.random.randn(2, 3, 5).astype("f"), backend="cpu")
    labels = Tensor(np.array([[1, -1, 3], [2, 0, -1]]), backend="cpu")

    result = objective.forward(scores, labels)
    dx = objective.backward()

    assert result.loss.shape == ()
    assert dx.shape == scores.shape
    np.testing.assert_array_equal(dx.data[0, 1, :], np.zeros(5))
    np.testing.assert_array_equal(dx.data[1, 2, :], np.zeros(5))
