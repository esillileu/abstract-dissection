import numpy as np
import pytest

from mlprosection import Tensor
from mlprosection.nn.layers.regulizer import Dropout


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
