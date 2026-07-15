# mlprosection/tensor/core.py

from __future__ import annotations

from typing import Any

from mlprosection.core.backend import Array, Backend, resolve_backend
from .mixins import (
    TensorOpsMixin,
    TensorDeviceMixin,
    TensorGradMixin,
    TensorIndexingMixin,
    TensorShapeMixin,
    TensorReductionMixin,
)


class Tensor(
    TensorOpsMixin,
    TensorDeviceMixin,
    TensorGradMixin,
    TensorIndexingMixin,
    TensorShapeMixin,
    TensorReductionMixin,
):
    __array_priority__ = 1000

    def __init__(
        self,
        data: Any,
        backend: Backend | str | None = None,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> None:
        if isinstance(data, Tensor):
            source = data

            if backend is None:
                self.backend = source.backend
                self.data = source.data
            else:
                self.backend = resolve_backend(backend)
                self.data = self.backend.move_array_from(
                    source.data,
                    source=source.backend,
                )

            self.grad = source.grad
            self.requires_grad = requires_grad or source.requires_grad
            self.creator = None
            self.name = name or source.name
            return

        self.backend = resolve_backend(backend)
        self.data = self.backend.asarray(data)
        self.grad: Array | None = None
        self.requires_grad = requires_grad
        self.creator: Any | None = None
        self.name = name

    @property
    def device(self) -> str:
        return self.backend.device

    @property
    def xp(self):
        return self.backend.xp

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        name = f", name={self.name!r}" if self.name is not None else ""
        return (
            f"Tensor(shape={self.shape}, dtype={self.dtype}, "
            f"device={self.device!r}, requires_grad={self.requires_grad}{name})"
        )
