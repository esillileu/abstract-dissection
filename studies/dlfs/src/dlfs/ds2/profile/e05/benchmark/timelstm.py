"""Isolated TimeLSTM layer forward and backward benchmarks."""

from __future__ import annotations

from deepscratch.core import Tensor
from deepscratch.nn.layers import TimeLSTM

from ..phase1 import Phase1TimeLSTM
from ..phase2 import Phase2TimeLSTM
from ..reference import ReferenceTimeLSTM
from .timing import _repeat


def benchmark_timelstm(
    backend,
    *,
    implementation: str,
    warmup: int,
    iterations: int,
    repetitions: int,
    shape: tuple[int, int, int, int] = (20, 35, 650, 650),
) -> dict[str, object]:
    n, time_size, input_size, hidden_size = shape
    xp = backend.xp
    cls = {
        "reference": ReferenceTimeLSTM,
        "production": TimeLSTM,
        "phase1": Phase1TimeLSTM,
        "phase2": Phase2TimeLSTM,
        "phase3": Phase2TimeLSTM,
    }[implementation]
    xp.random.seed(20260811)
    layer = cls(input_size, hidden_size, stateful=True, backend=backend)
    xs = Tensor(
        xp.random.randn(n, time_size, input_size).astype(xp.float32), backend=backend
    )
    dhs = Tensor(
        xp.random.randn(n, time_size, hidden_size).astype(xp.float32), backend=backend
    )
    h0 = xp.random.randn(n, hidden_size).astype(xp.float32)
    c0 = xp.random.randn(n, hidden_size).astype(xp.float32)

    def forward() -> None:
        layer.set_state(h0, c0)
        layer.forward(xs)

    forward_result = _repeat(
        forward, backend, warmup=warmup, iterations=iterations, repetitions=repetitions
    )
    layer.set_state(h0, c0)
    layer.forward(xs)
    backward_result = _repeat(
        lambda: layer.backward(dhs),
        backend,
        warmup=warmup,
        iterations=iterations,
        repetitions=repetitions,
    )
    return {
        "shape": {"N": n, "T": time_size, "D": input_size, "H": hidden_size},
        "forward": forward_result,
        "backward": backward_result,
    }
