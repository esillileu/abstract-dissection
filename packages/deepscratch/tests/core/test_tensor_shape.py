import numpy as np
from deepscratch.core import Tensor


def test_astype_changes_dtype():
    x = Tensor([1, 2, 3])

    y = x.astype(x.backend.float_dtype)

    assert y.dtype == x.backend.float_dtype
    assert y.requires_grad == x.requires_grad


def test_reshape_with_args():
    x = Tensor([1, 2, 3, 4])

    y = x.reshape(2, 2)

    assert y.shape == (2, 2)
    np.testing.assert_array_equal(y.numpy(), np.array([[1, 2], [3, 4]]))


def test_reshape_with_tuple():
    x = Tensor([1, 2, 3, 4])

    y = x.reshape((2, 2))

    assert y.shape == (2, 2)
    np.testing.assert_array_equal(y.numpy(), np.array([[1, 2], [3, 4]]))


def test_flatten():
    x = Tensor([[1, 2], [3, 4]])

    y = x.flatten()

    assert y.shape == (4,)
    np.testing.assert_array_equal(y.numpy(), np.array([1, 2, 3, 4]))


def test_transpose_without_axes():
    x = Tensor([[1, 2, 3], [4, 5, 6]])

    y = x.transpose()

    assert y.shape == (3, 2)
    np.testing.assert_array_equal(y.numpy(), np.array([[1, 4], [2, 5], [3, 6]]))


def test_transpose_with_axes():
    x = Tensor(np.arange(24).reshape(2, 3, 4))

    y = x.transpose(1, 0, 2)

    assert y.shape == (3, 2, 4)
    np.testing.assert_array_equal(
        y.numpy(), np.transpose(np.arange(24).reshape(2, 3, 4), (1, 0, 2))
    )


def test_T_property():
    x = Tensor([[1, 2, 3], [4, 5, 6]])

    y = x.T

    assert y.shape == (3, 2)
    np.testing.assert_array_equal(y.numpy(), np.array([[1, 4], [2, 5], [3, 6]]))
