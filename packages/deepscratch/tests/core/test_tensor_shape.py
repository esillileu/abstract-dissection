import numpy as np
from deepscratch.core import Tensor


def test_tensor_shape_and_reshape():
    x = Tensor([1, 2, 3])
    y = x.astype(x.backend.float_dtype)
    assert y.dtype == x.backend.float_dtype
    assert y.requires_grad == x.requires_grad

    data = Tensor([1, 2, 3, 4])
    np.testing.assert_array_equal(
        data.reshape(2, 2).numpy(), np.array([[1, 2], [3, 4]])
    )
    np.testing.assert_array_equal(
        data.reshape((2, 2)).numpy(), np.array([[1, 2], [3, 4]])
    )

    matrix = Tensor([[1, 2], [3, 4]])
    np.testing.assert_array_equal(matrix.flatten().numpy(), np.array([1, 2, 3, 4]))


def test_tensor_transpose():
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    expected = np.array([[1, 4], [2, 5], [3, 6]])
    np.testing.assert_array_equal(x.transpose().numpy(), expected)
    np.testing.assert_array_equal(x.T.numpy(), expected)

    arr = np.arange(24).reshape(2, 3, 4)
    t = Tensor(arr).transpose(1, 0, 2)
    assert t.shape == (3, 2, 4)
    np.testing.assert_array_equal(t.numpy(), np.transpose(arr, (1, 0, 2)))
