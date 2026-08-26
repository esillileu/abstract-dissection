from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepscratch.core.backend import assert_same_device

if TYPE_CHECKING:
    from ..base import Tensor


class TensorOpsMixin:
    def __add__(self: Tensor, other: Any) -> Tensor:
        from ..creation import as_tensor

        other = as_tensor(other, backend=self.backend)
        assert_same_device(self, other)

        return type(self)(
            self.data + other.data,
            backend=self.backend,
            requires_grad=self.requires_grad or other.requires_grad,
        )

    def __radd__(self: Tensor, other: Any) -> Tensor:
        return self.__add__(other)

    def __sub__(self: Tensor, other: Any) -> Tensor:
        from ..creation import as_tensor

        other = as_tensor(other, backend=self.backend)
        assert_same_device(self, other)

        return type(self)(
            self.data - other.data,
            backend=self.backend,
            requires_grad=self.requires_grad or other.requires_grad,
        )

    def __rsub__(self: Tensor, other: Any) -> Tensor:
        from ..creation import as_tensor

        other = as_tensor(other, backend=self.backend)
        assert_same_device(self, other)

        return type(self)(
            other.data - self.data,
            backend=self.backend,
            requires_grad=self.requires_grad or other.requires_grad,
        )

    def __mul__(self: Tensor, other: Any) -> Tensor:
        from ..creation import as_tensor

        other = as_tensor(other, backend=self.backend)
        assert_same_device(self, other)

        return type(self)(
            self.data * other.data,
            backend=self.backend,
            requires_grad=self.requires_grad or other.requires_grad,
        )

    def __rmul__(self: Tensor, other: Any) -> Tensor:
        return self.__mul__(other)

    def __truediv__(self: Tensor, other: Any) -> Tensor:
        from ..creation import as_tensor

        other = as_tensor(other, backend=self.backend)
        assert_same_device(self, other)

        return type(self)(
            self.data / other.data,
            backend=self.backend,
            requires_grad=self.requires_grad or other.requires_grad,
        )

    def __rtruediv__(self: Tensor, other: Any) -> Tensor:
        from ..creation import as_tensor

        other = as_tensor(other, backend=self.backend)
        assert_same_device(self, other)

        return type(self)(
            other.data / self.data,
            backend=self.backend,
            requires_grad=self.requires_grad or other.requires_grad,
        )

    def __neg__(self: Tensor) -> Tensor:

        return type(self)(
            -self.data,
            backend=self.backend,
            requires_grad=self.requires_grad,
        )

    def __matmul__(self: Tensor, other: Any) -> Tensor:
        from ..creation import as_tensor

        other = as_tensor(other, backend=self.backend)
        assert_same_device(self, other)

        return type(self)(
            self.data @ other.data,
            backend=self.backend,
            requires_grad=self.requires_grad or other.requires_grad,
        )

    def __rmatmul__(self: Tensor, other: Any) -> Tensor:
        from ..creation import as_tensor

        other = as_tensor(other, backend=self.backend)
        assert_same_device(self, other)

        return type(self)(
            other.data @ self.data,
            backend=self.backend,
            requires_grad=self.requires_grad or other.requires_grad,
        )

    def __le__(self: Tensor, other: Any) -> Tensor:
        from ..creation import as_tensor

        other = as_tensor(other, backend=self.backend)
        assert_same_device(self, other)

        return type(self)(
            self.data <= other.data,
            backend=self.backend,
            requires_grad=False,
        )
