import numpy as np
import pytest

from mlprosection.core.tensor import Tensor


def test_cpu_returns_self_when_already_cpu():
    x = Tensor([1, 2, 3], backend="cpu")

    y = x.cpu()

    assert y is x


def test_to_same_device_returns_self():
    x = Tensor([1, 2, 3], backend="cpu")

    y = x.to("cpu")

    assert y is x


def test_numpy_returns_numpy_array():
    x = Tensor([1, 2, 3], backend="cpu")

    arr = x.numpy()

    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, np.array([1, 2, 3]))


def test_item_returns_python_scalar():
    x = Tensor(3.14, backend="cpu")

    value = x.item()

    assert isinstance(value, float)
    assert value == pytest.approx(3.14)


def test_to_gpu_optional():
    _ = pytest.importorskip("cupy")

    x = Tensor([1, 2, 3], backend="cpu")

    try:
        y = x.gpu()
    except Exception as exc:
        pytest.skip(f"GPU backend is not available: {exc}")

    assert y.device.startswith("cuda:")
    np.testing.assert_array_equal(y.cpu().numpy(), np.array([1, 2, 3]))