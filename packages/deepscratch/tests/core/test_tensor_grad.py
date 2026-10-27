import numpy as np
from deepscratch.core import Tensor


def test_tensor_set_and_zero_grad():
    x = Tensor([1.0, 2.0, 3.0])
    x.set_grad([0.1, 0.2, 0.3])
    np.testing.assert_allclose(x.grad, np.array([0.1, 0.2, 0.3]))

    g = Tensor([0.4, 0.5, 0.6])
    x.set_grad(g)
    assert x.grad is g.data

    x.zero_grad()
    np.testing.assert_allclose(x.grad, np.array([0.0, 0.0, 0.0]))

    x.set_grad(None)
    assert x.grad is None
    x.zero_grad()
    assert x.grad is None


def test_tensor_detach_and_copy():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True, name="x")
    y = x.detach()
    assert isinstance(y, Tensor) and y is not x
    assert y.data is x.data
    assert y.requires_grad is False
    assert y.name == "x"

    x.set_grad([0.1, 0.2, 0.3])
    c = x.copy()
    assert c is not x
    np.testing.assert_allclose(c.data, x.data)
    np.testing.assert_allclose(c.grad, x.grad)
    assert c.requires_grad == x.requires_grad
    assert c.name == "x"
