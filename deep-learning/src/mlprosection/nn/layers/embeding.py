from __future__ import annotations

from mlprosection import Tensor

from .base import Layer
from ..types import Parameter


class Embedding(Layer):
    def __init__(self, vocab_length: int, dim: int):
        self.W = Parameter(Tensor.randn(vocab_length, dim))
        self.idx = None

    def forward(self, idx):
        xp = self.W.backend.xp
        self.idx = xp.asarray(idx, dtype=xp.int64)
        return self.W[self.idx]

    def backward(self, dout: Tensor) -> None:
        xp = self.W.backend.xp

        if self.W.grad is None:
            self.W.grad = xp.zeros_like(self.W.data)

        xp.add.at(self.W.grad, self.idx, dout.data)

        return None
