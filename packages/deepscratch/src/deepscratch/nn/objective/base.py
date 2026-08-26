"""Loss/objective contracts independent from prediction models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deepscratch.core import Tensor
from deepscratch.nn.layers import Layer


@dataclass(frozen=True)
class ObjectiveResult:
    """An optimized loss plus optional standardized reporting projection.

    ``loss`` is the scalar whose gradient is cached for ``backward``.
    ``reporting_loss`` lets objectives preserve the repository-wide mean-loss
    metric when the optimized source objective intentionally uses another
    reduction, such as the book Word2Vec sum-over-terms loss.
    """

    loss: Tensor
    unit_count: int
    replay_context: Any = None
    reporting_loss: Tensor | None = None


class Objective(Layer):
    """An objective that caches the state required for its backward pass."""

    def forward_manual(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        cache: bool = True,
        replay_context: Any = None,
    ) -> ObjectiveResult:
        raise NotImplementedError

    def backward_manual(self) -> Tensor:
        raise NotImplementedError
