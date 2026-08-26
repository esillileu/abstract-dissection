from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..nn.types import NamedParameters


class OptimizerTransform(Protocol):
    def __call__(
        self,
        params: NamedParameters,
    ) -> None: ...


class L2Regularization:
    def __init__(self, coefficient: float = 1e-4) -> None:
        if coefficient < 0.0:
            raise ValueError("coefficient must be non-negative")

        self.coefficient = coefficient

    def __call__(self, params: NamedParameters) -> None:
        for _, param in params:
            if param.grad is None or param.ndim < 2:
                continue

            param.grad += self.coefficient * param.data


class ClipGradNorm:
    def __init__(self, max_norm: float = 5, eps: float = 1e-6) -> None:
        if max_norm <= 0.0:
            raise ValueError("max_norm must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.max_norm = max_norm
        self.eps = eps

    def __call__(self, params: NamedParameters) -> None:
        if not params:
            return

        first_grad = params[0][1].grad
        if first_grad is None:
            return
        xp = params[0][1].backend.xp
        total_sq = xp.asarray(0.0, dtype=first_grad.dtype)

        for _, param in params:
            if param.grad is None:
                continue

            total_sq += (param.grad**2).sum()

        total_norm = xp.sqrt(total_sq)
        scale = self.max_norm / (total_norm + self.eps)
        one = xp.asarray(1.0, dtype=total_norm.dtype)
        scale = xp.where(scale < one, scale, one)

        for _, param in params:
            if param.grad is not None:
                param.grad *= scale
