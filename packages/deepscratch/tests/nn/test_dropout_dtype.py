import numpy as np
import pytest
from deepscratch.core import Tensor
from deepscratch.nn.layers.regulizer import Dropout


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("inverted", [False, True])
def test_dropout_preserves_input_dtype_in_train_and_eval(dtype, inverted):
    layer = Dropout(0.5, inverted=inverted)
    x = Tensor(np.ones((4, 4), dtype=dtype), backend="cpu")

    train = layer.forward(x)
    layer.eval()
    evaluation = layer.forward(x)

    assert train.dtype == dtype
    assert evaluation.dtype == dtype


def test_inverted_dropout_uses_typed_scale_without_changing_values():
    np.random.seed(7)
    layer = Dropout(0.5, inverted=True)
    x = Tensor(np.ones((8, 8), dtype=np.float32), backend="cpu")

    output = layer.forward(x)

    assert set(np.unique(output.data)) <= {0.0, 2.0}


@pytest.mark.parametrize(
    ("inverted", "expected_scale"),
    [(False, 1.0), (True, 2.0)],
)
def test_dropout_backward_uses_the_forward_mask_and_scale(inverted, expected_scale):
    np.random.seed(7)
    layer = Dropout(0.5, inverted=inverted)
    x = Tensor(np.ones((8, 8), dtype=np.float32), backend="cpu")
    output = layer.forward(x)

    gradient = layer.backward(Tensor(np.ones_like(output.data), backend="cpu"))

    expected = (output.data != 0).astype(np.float32) * expected_scale
    np.testing.assert_array_equal(gradient.data, expected)
    assert gradient.dtype == np.float32


def test_dropout_backward_requires_a_training_forward():
    layer = Dropout(0.5, inverted=True)

    with pytest.raises(RuntimeError, match="training forward"):
        layer.backward(Tensor(np.ones((2, 2), dtype=np.float32), backend="cpu"))
