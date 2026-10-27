import numpy as np
import pytest
from deepscratch.core import Tensor


def test_tensor_device_and_conversion():
    x = Tensor([1, 2, 3], backend="cpu")
    assert x.cpu() is x
    assert x.to("cpu") is x

    arr = x.numpy()
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, np.array([1, 2, 3]))

    scalar = Tensor(3.14, backend="cpu")
    assert isinstance(scalar.item(), float)
    assert scalar.item() == pytest.approx(3.14)


def test_to_gpu_optional():
    _ = pytest.importorskip("cupy")
    x = Tensor([1, 2, 3], backend="cpu")
    try:
        y = x.gpu()
    except Exception as exc:
        pytest.skip(f"GPU backend is not available: {exc}")
    assert y.device.startswith("cuda:")
