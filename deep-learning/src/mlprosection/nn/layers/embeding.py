"""Index embedding layer usable directly or through ``TimeDistributed``."""

from __future__ import annotations

from typing import Any

from mlprosection import Tensor
from mlprosection.core.backend import Backend, resolve_backend

from .base import Layer
from ..types import Parameter


class Embedding(Layer):
    def __init__(self, vocab_length: int, dim: int, *, backend: Backend | None = None, weight: Parameter | None = None) -> None:
        target = backend or (weight.backend if weight is not None else None)
        super().__init__(resolve_backend(target) if target is not None else None)
        self.W = weight or Parameter(0.01 * self._backend.xp.random.randn(vocab_length, dim).astype(self._backend.float_dtype), backend=self._backend, name="W")
        self.idx = None

    def forward_manual(self, idx: Tensor | Any) -> Tensor:
        xp = self.W.backend.xp
        values = idx.data if isinstance(idx, Tensor) else idx
        self.idx = xp.asarray(values, dtype=xp.int64)
        return Tensor(self.W.data[self.idx], backend=self.W.backend)

    def backward_manual(self, dout: Tensor) -> None:
        if self.idx is None:
            raise RuntimeError("forward() must be called before backward()")
        xp = self.W.backend.xp
        self.W.grad.fill(0)
        xp.add.at(self.W.grad, self.idx, dout.data)
        return None
