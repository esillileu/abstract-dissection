from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from mlprosection.nn.types import Parameter


def clip_grads(named_params: Sequence[tuple[str, Parameter]], max_norm):
    params = [param for _, param in named_params]
    if not params:
        return

    total_norm = 0
    xp = params[0].backend.xp

    for param in params:
        total_norm += xp.sum(param.grad**2)
    total_norm = xp.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for param in params:
            param.grad *= rate
