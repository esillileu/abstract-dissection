import numpy as np

from mlprosection.core.tensor import Tensor


def test_sum_all():
    x = Tensor([[1, 2], [3, 4]])

    y = x.sum()

    assert y.shape == ()
    assert y.item() == 10


def test_sum_axis_keepdims():
    x = Tensor([[1, 2], [3, 4]])

    y = x.sum(axis=1, keepdims=True)

    assert y.shape == (2, 1)
    np.testing.assert_array_equal(y.numpy(), np.array([[3], [7]]))


def test_mean_all():
    x = Tensor([[1, 2], [3, 4]])

    y = x.mean()

    assert y.shape == ()
    assert y.item() == 2.5


def test_mean_axis():
    x = Tensor([[1, 2], [3, 4]])

    y = x.mean(axis=0)

    np.testing.assert_allclose(y.numpy(), np.array([2.0, 3.0]))


def test_max_axis():
    x = Tensor([[1, 5], [3, 4]])

    y = x.max(axis=1)

    np.testing.assert_array_equal(y.numpy(), np.array([5, 4]))


def test_argmax_axis_does_not_require_grad():
    x = Tensor([[1, 5], [3, 4]], requires_grad=True)

    y = x.argmax(axis=1)

    assert y.requires_grad is False
    np.testing.assert_array_equal(y.numpy(), np.array([1, 1]))