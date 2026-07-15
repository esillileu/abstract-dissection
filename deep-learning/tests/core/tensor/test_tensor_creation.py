import numpy as np

from mlprosection.core.tensor import Tensor, as_tensor, tensor


def test_tensor_creation_from_list():
    x = Tensor([1, 2, 3])

    assert x.shape == (3,)
    assert x.ndim == 1
    assert x.size == 3
    assert x.requires_grad is False
    assert x.grad is None


def test_tensor_factory_function():
    x = tensor([1, 2, 3], requires_grad=True, name="x")

    assert isinstance(x, Tensor)
    assert x.requires_grad is True
    assert x.name == "x"
    np.testing.assert_array_equal(x.numpy(), np.array([1, 2, 3]))


def test_as_tensor_returns_same_tensor():
    x = Tensor([1, 2, 3])
    y = as_tensor(x)

    assert y is x


def test_tensor_copy_constructor_keeps_data_and_grad_flag():
    x = Tensor([1, 2, 3], requires_grad=True, name="x")
    y = Tensor(x)

    assert y is not x
    assert y.data is x.data
    assert y.backend is x.backend
    assert y.requires_grad is True
    assert y.name == "x"


def test_tensor_repr_contains_basic_info():
    x = Tensor([1, 2, 3], requires_grad=True, name="x")
    text = repr(x)

    assert "Tensor" in text
    assert "shape=" in text
    assert "dtype=" in text
    assert "device=" in text
    assert "requires_grad=True" in text
    assert "name='x'" in text