"""Loss/objective contracts independent from prediction models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlprosection import Tensor
from mlprosection.nn.layers import Layer


@dataclass(frozen=True)
class ObjectiveResult:
    loss: Tensor
    unit_count: int
    replay_context: Any = None


class Objective(Layer):
    """A trainable objective that caches its own backward state."""

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
