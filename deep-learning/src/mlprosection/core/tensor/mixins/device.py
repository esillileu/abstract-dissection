from __future__ import annotations

from typing import Any, TYPE_CHECKING

from mlprosection.core.backend import Array, Backend, resolve_backend

if TYPE_CHECKING:
    from mlprosection.core.tensor.base import Tensor


class TensorDeviceMixin:
    def to(self: Tensor, target: Backend | str) -> Tensor:
        backend = resolve_backend(target)

        if backend.device == self.device:
            return self

        data = backend.move_array_from(self.data, source=self.backend)

        out = type(self)(
            data,
            backend=backend,
            requires_grad=self.requires_grad,
            name=self.name,
        )

        if self.grad is not None:
            out.grad = backend.move_array_from(self.grad, source=self.backend)

        return out

    def cpu(self: Tensor) -> Tensor:
        return self.to("cpu")

    def gpu(self: Tensor, device: str = "cuda:0") -> Tensor:
        return self.to(device)

    def numpy(self: Tensor) -> Array:
        return self.backend.to_numpy(self.data)

    def item(self: Tensor) -> Any:
        return self.backend.to_numpy(self.data).item()
