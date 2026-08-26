import numpy as np
from deepscratch.core import Tensor


def test_set_grad_from_array():
    x = Tensor([1.0, 2.0, 3.0])
    x.set_grad([0.1, 0.2, 0.3])

    np.testing.assert_allclose(x.grad, np.array([0.1, 0.2, 0.3]))


def test_set_grad_from_tensor():
    x = Tensor([1.0, 2.0, 3.0])
    g = Tensor([0.1, 0.2, 0.3])

    x.set_grad(g)

    assert x.grad is g.data


def test_set_grad_none_clears_grad():
    x = Tensor([1.0, 2.0, 3.0])
    x.set_grad([1.0, 1.0, 1.0])

    x.set_grad(None)

    assert x.grad is None


def test_zero_grad_sets_existing_grad_to_zero():
    x = Tensor([1.0, 2.0, 3.0])
    x.set_grad([1.0, 2.0, 3.0])

    x.zero_grad()

    np.testing.assert_allclose(x.grad, np.array([0.0, 0.0, 0.0]))


def test_zero_grad_without_grad_does_nothing():
    x = Tensor([1.0, 2.0, 3.0])

    x.zero_grad()

    assert x.grad is None


def test_detach_returns_tensor_without_grad_tracking():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True, name="x")

    y = x.detach()

    assert isinstance(y, Tensor)
    assert y is not x
    assert y.data is x.data
    assert y.requires_grad is False
    assert y.name == "x"


def test_copy_copies_data_and_grad():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True, name="x")
    x.set_grad([0.1, 0.2, 0.3])

    y = x.copy()

    assert y is not x
    assert y.data is not x.data
    assert y.grad is not x.grad
    assert y.requires_grad is True
    assert y.name == "x"

    np.testing.assert_allclose(y.numpy(), x.numpy())
    np.testing.assert_allclose(y.grad, x.grad)
