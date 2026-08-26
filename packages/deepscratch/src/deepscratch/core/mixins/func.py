from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import Tensor


class TensorFuncMixin:
    def exp(self: Tensor) -> Tensor:
        return type(self)(
            self.backend.xp.exp(self.data),
            backend=self.backend,
            requires_grad=False,
        )

    def log(self: Tensor) -> Tensor:
        return type(self)(
            self.backend.xp.log(self.data),
            backend=self.backend,
            requires_grad=False,
        )

    def sqrt(self: Tensor) -> Tensor:
        return type(self)(
            self.backend.xp.sqrt(self.data),
            backend=self.backend,
            requires_grad=False,
        )
