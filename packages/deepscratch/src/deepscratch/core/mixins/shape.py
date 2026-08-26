from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..base import Tensor


class TensorShapeMixin:
    def astype(self: Tensor, dtype: Any) -> Tensor:
        return type(self)(
            self.data.astype(dtype),
            backend=self.backend,
            requires_grad=self.requires_grad,
            name=self.name,
        )

    def reshape(self: Tensor, *shape: int | tuple[int, ...]) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]

        return type(self)(
            self.data.reshape(*shape),
            backend=self.backend,
            requires_grad=self.requires_grad,
        )

    def flatten(self: Tensor) -> Tensor:
        return self.reshape(-1)

    def transpose(self: Tensor, *axes: int | tuple[int, ...]) -> Tensor:
        if len(axes) == 1 and isinstance(axes[0], tuple):
            axes = axes[0]

        if len(axes) == 0:
            data = self.data.transpose()
        else:
            data = self.data.transpose(*axes)

        return type(self)(
            data,
            backend=self.backend,
            requires_grad=self.requires_grad,
        )

    @property
    def T(self: Tensor) -> Tensor:
        return type(self)(
            self.data.T,
            backend=self.backend,
            requires_grad=self.requires_grad,
        )
