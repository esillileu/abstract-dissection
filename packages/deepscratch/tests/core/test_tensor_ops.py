import numpy as np
from deepscratch.core import Tensor


def test_add_tensor():
    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])

    z = x + y

    np.testing.assert_array_equal(z.numpy(), np.array([5, 7, 9]))


def test_add_scalar():
    x = Tensor([1, 2, 3])

    z = x + 10

    np.testing.assert_array_equal(z.numpy(), np.array([11, 12, 13]))


def test_radd_scalar():
    x = Tensor([1, 2, 3])

    z = 10 + x

    np.testing.assert_array_equal(z.numpy(), np.array([11, 12, 13]))


def test_sub_tensor():
    x = Tensor([5, 6, 7])
    y = Tensor([1, 2, 3])

    z = x - y

    np.testing.assert_array_equal(z.numpy(), np.array([4, 4, 4]))


def test_rsub_scalar():
    x = Tensor([1, 2, 3])

    z = 10 - x

    np.testing.assert_array_equal(z.numpy(), np.array([9, 8, 7]))


def test_mul_tensor():
    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])

    z = x * y

    np.testing.assert_array_equal(z.numpy(), np.array([4, 10, 18]))


def test_rmul_scalar():
    x = Tensor([1, 2, 3])

    z = 2 * x

    np.testing.assert_array_equal(z.numpy(), np.array([2, 4, 6]))


def test_div_tensor():
    x = Tensor([2.0, 4.0, 8.0])
    y = Tensor([2.0, 2.0, 4.0])

    z = x / y

    np.testing.assert_allclose(z.numpy(), np.array([1.0, 2.0, 2.0]))


def test_rdiv_scalar():
    x = Tensor([2.0, 4.0, 8.0])

    z = 16.0 / x

    np.testing.assert_allclose(z.numpy(), np.array([8.0, 4.0, 2.0]))


def test_neg():
    x = Tensor([1, -2, 3])

    z = -x

    np.testing.assert_array_equal(z.numpy(), np.array([-1, 2, -3]))


def test_matmul():
    x = Tensor([[1, 2], [3, 4]])
    y = Tensor([[10], [20]])

    z = x @ y

    np.testing.assert_array_equal(z.numpy(), np.array([[50], [110]]))


def test_rmatmul():
    x = Tensor([[1, 2], [3, 4]])

    z = np.array([[10, 20]]) @ x

    np.testing.assert_array_equal(z.numpy(), np.array([[70, 100]]))


def test_requires_grad_propagates_for_binary_ops():
    x = Tensor([1, 2, 3], requires_grad=True)
    y = Tensor([4, 5, 6], requires_grad=False)

    z = x + y

    assert z.requires_grad is True


def test_requires_grad_false_when_both_false():
    x = Tensor([1, 2, 3], requires_grad=False)
    y = Tensor([4, 5, 6], requires_grad=False)

    z = x + y

    assert z.requires_grad is False
