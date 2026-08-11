from __future__ import annotations

import numpy as np
import pytest

from mlprosection import Tensor
from mlprosection.core.backend import BackendConfig, make_backend

from exp.ds2.profile.e05.validation import compare_lstm, error_metrics


@pytest.mark.parametrize("shape", [(1, 1, 1, 1), (2, 3, 4, 5), (3, 2, 5, 4)])
def test_phase1_timelstm_matches_reference_on_cpu(shape):
    backend = make_backend(BackendConfig(device="cpu", dtype="float32", seed=0))
    assert compare_lstm(backend, shape, seed=7)["passed"]


def test_error_metrics_normalizes_against_both_tensors():
    backend = make_backend(BackendConfig(device="cpu", dtype="float32"))
    result = error_metrics(
        backend,
        np.asarray([0.0, 2.0], dtype=np.float32),
        np.asarray([1e-7, 1.0], dtype=np.float32),
    )
    assert result["max_absolute"] == 1.0
    assert result["max_relative"] == pytest.approx(0.5)


def test_cache_false_retains_state_but_rejects_backward():
    backend = make_backend(BackendConfig(device="cpu", dtype="float32", seed=3))
    from mlprosection.nn.layers import TimeLSTM

    layer = TimeLSTM(2, 3, stateful=True, backend=backend)
    output = layer.forward(Tensor(np.ones((2, 2, 2), dtype=np.float32), backend=backend), cache=False)
    assert layer.h is not None and layer.c is not None and layer.layers == []
    with pytest.raises(RuntimeError, match="forward must be called"):
        layer.backward(Tensor(np.ones_like(output.data), backend=backend))
