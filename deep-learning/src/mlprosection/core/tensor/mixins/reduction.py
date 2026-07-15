from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlprosection.core.tensor.base import Tensor


class TensorReductionMixin:
    def sum(
        self: Tensor, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Tensor:
        return type(self)(
            self.data.sum(axis=axis, keepdims=keepdims),
            backend=self.backend,
            requires_grad=self.requires_grad,
        )

    def mean(
        self: Tensor, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Tensor:
        return type(self)(
            self.data.mean(axis=axis, keepdims=keepdims),
            backend=self.backend,
            requires_grad=self.requires_grad,
        )

    def max(
        self: Tensor, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Tensor:
        return type(self)(
            self.data.max(axis=axis, keepdims=keepdims),
            backend=self.backend,
            requires_grad=self.requires_grad,
        )

    def argmax(self: Tensor, axis: int | None = None) -> Tensor:
        return type(self)(
            self.data.argmax(axis=axis),
            backend=self.backend,
            requires_grad=False,
        )
