import numpy as np

from mlprosection.core.tensor import Tensor


def test_getitem_single_index():
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)

    y = x[0]

    assert y.shape == (2,)
    assert y.requires_grad is True
    np.testing.assert_array_equal(y.numpy(), np.array([1, 2]))


def test_getitem_slice():
    x = Tensor([1, 2, 3, 4])

    y = x[1:3]

    np.testing.assert_array_equal(y.numpy(), np.array([2, 3]))


def test_setitem_scalar():
    x = Tensor([1, 2, 3])

    x[1] = 99

    np.testing.assert_array_equal(x.numpy(), np.array([1, 99, 3]))


def test_setitem_tensor():
    x = Tensor([1, 2, 3])
    y = Tensor([8, 9])

    x[1:] = y

    np.testing.assert_array_equal(x.numpy(), np.array([1, 8, 9]))