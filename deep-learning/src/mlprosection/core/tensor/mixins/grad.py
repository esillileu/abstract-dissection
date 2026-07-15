from __future__ import annotations

from typing import TYPE_CHECKING

from mlprosection.core.backend import Array, assert_same_device

if TYPE_CHECKING:
    from mlprosection.core.tensor.base import Tensor


class TensorGradMixin:
    def zero_grad(self: Tensor) -> None:
        if self.grad is not None:
            self.grad[...] = 0

    def set_grad(self: Tensor, grad: Tensor | Array | None) -> None:
        if grad is None:
            self.grad = None
            return

        if isinstance(grad, type(self)):
            assert_same_device(self, grad)
            self.grad = grad.data
            return

        self.grad = self.backend.asarray(grad)

    def detach(self: Tensor) -> Tensor:
        return type(self)(
            self.data,
            backend=self.backend,
            requires_grad=False,
            name=self.name,
        )

    def copy(self: Tensor) -> Tensor:
        out = type(self)(
            self.data.copy(),
            backend=self.backend,
            requires_grad=self.requires_grad,
            name=self.name,
        )

        if self.grad is not None:
            out.grad = self.grad.copy()

        return out
