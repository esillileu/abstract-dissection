"""Layer-level comparison between reference and optimized TimeLSTM."""

from __future__ import annotations

from deepscratch.core import Tensor
from deepscratch.nn.layers import TimeLSTM

from ..phase1 import Phase1TimeLSTM
from ..phase2 import Phase2TimeLSTM
from ..reference import ReferenceTimeLSTM
from .metrics import FORWARD_CEILING, GRADIENT_CEILING, _within, error_metrics


def _copy_lstm(source, target) -> None:
    target.Wx.data[...] = source.Wx.data
    target.Wh.data[...] = source.Wh.data
    target.b.data[...] = source.b.data


def compare_lstm(
    backend,
    shape: tuple[int, int, int, int],
    seed: int,
    *,
    implementation: str = "phase1",
):
    n, time_size, input_size, hidden_size = shape
    xp = backend.xp
    xp.random.seed(seed)
    reference = ReferenceTimeLSTM(
        input_size, hidden_size, stateful=True, backend=backend
    )
    actual_cls = {
        "phase1": Phase1TimeLSTM,
        "phase2": Phase2TimeLSTM,
        "phase3": Phase2TimeLSTM,
    }.get(implementation, TimeLSTM)
    actual = actual_cls(input_size, hidden_size, stateful=True, backend=backend)
    _copy_lstm(reference, actual)
    xs = Tensor(
        xp.random.randn(n, time_size, input_size).astype(xp.float32), backend=backend
    )
    dhs = Tensor(
        xp.random.randn(n, time_size, hidden_size).astype(xp.float32), backend=backend
    )
    h0 = xp.random.randn(n, hidden_size).astype(xp.float32)
    c0 = xp.random.randn(n, hidden_size).astype(xp.float32)
    reference.set_state(h0, c0)
    actual.set_state(h0, c0)
    expected = reference.forward(xs)
    observed = actual.forward(xs)
    outputs = {
        "output": error_metrics(backend, expected.data, observed.data),
        "final_h": error_metrics(backend, reference.h, actual.h),
        "final_c": error_metrics(backend, reference.c, actual.c),
    }
    expected_dx = reference.backward(dhs)
    observed_dx = actual.backward(dhs)
    gradients = {
        "dx": error_metrics(backend, expected_dx.data, observed_dx.data),
        "dWx": error_metrics(backend, reference.Wx.grad, actual.Wx.grad),
        "dWh": error_metrics(backend, reference.Wh.grad, actual.Wh.grad),
        "db": error_metrics(backend, reference.b.grad, actual.b.grad),
        "dh": error_metrics(backend, reference.dh.data, actual.dh.data),
    }

    # Consecutive stateful forward and lifecycle contracts.
    next_x = Tensor(
        xp.random.randn(n, 2, input_size).astype(xp.float32), backend=backend
    )
    outputs["stateful_next"] = error_metrics(
        backend, reference.forward(next_x).data, actual.forward(next_x).data
    )
    reference.detach_state()
    actual.detach_state()
    assert reference.layers == [] and actual.layers == []
    reference.reset_state()
    actual.reset_state()
    assert (
        reference.h is None
        and actual.h is None
        and reference.c is None
        and actual.c is None
    )
    no_cache = actual.forward(next_x, cache=False)
    assert no_cache.shape == (n, 2, hidden_size) and actual.layers == []
    try:
        actual.backward(Tensor(xp.zeros_like(no_cache.data), backend=backend))
    except RuntimeError:
        pass
    else:
        raise AssertionError("cache=False unexpectedly allowed backward")
    changed = Tensor(
        xp.random.randn(n + 1, 1, input_size).astype(xp.float32), backend=backend
    )
    actual.forward(changed, cache=False)
    assert actual.h.shape[0] == n + 1 and actual.c.shape[0] == n + 1

    passed = all(_within(value, FORWARD_CEILING) for value in outputs.values())
    passed &= all(_within(value, GRADIENT_CEILING) for value in gradients.values())
    return {"outputs": outputs, "gradients": gradients, "passed": bool(passed)}
