from __future__ import annotations

import numpy as np
import pytest
from deepscratch.core import Tensor
from deepscratch.core.backend import BackendConfig, make_backend
from deepscratch.nn.layers import TimeLSTM

cp = pytest.importorskip("cupy")


def has_cuda_device() -> bool:
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


pytestmark = pytest.mark.skipif(
    not has_cuda_device(),
    reason="CUDA device is not available.",
)


def test_fused_time_lstm_backward_accepts_noncontiguous_gradient() -> None:
    backend = make_backend(BackendConfig(device="cuda", dtype="float32", seed=3))
    rng = np.random.RandomState(7)
    xs = backend.asarray(rng.randn(6, 5, 9).astype("float32"))
    initial_h = backend.asarray(rng.randn(6, 7).astype("float32"))
    wide_gradient = backend.asarray(rng.randn(6, 5, 14).astype("float32"))
    gradient = wide_gradient[:, :, 7:]
    assert not gradient.flags.c_contiguous

    fused = TimeLSTM(9, 7, stateful=True, backend=backend)
    reference = TimeLSTM(9, 7, stateful=True, backend=backend)
    reference._fused_cuda = False
    for (_, fused_parameter), (_, reference_parameter) in zip(
        fused.named_parameters(), reference.named_parameters(), strict=True
    ):
        reference_parameter.data[...] = fused_parameter.data

    def run(layer: TimeLSTM):
        layer.set_state(Tensor(initial_h, backend=backend))
        layer.forward(Tensor(xs, backend=backend))
        dx = layer.backward(Tensor(gradient, backend=backend))
        parameter_grads = {
            name: backend.to_numpy(parameter.grad)
            for name, parameter in layer.named_parameters()
        }
        return (
            backend.to_numpy(dx.data),
            backend.to_numpy(layer.dh.data),
            parameter_grads,
        )

    fused_dx, fused_dh, fused_grads = run(fused)
    reference_dx, reference_dh, reference_grads = run(reference)

    np.testing.assert_allclose(fused_dx, reference_dx, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(fused_dh, reference_dh, rtol=2e-6, atol=2e-6)
    for name in fused_grads:
        np.testing.assert_allclose(
            fused_grads[name], reference_grads[name], rtol=2e-6, atol=2e-6
        )
