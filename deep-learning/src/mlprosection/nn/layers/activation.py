from __future__ import annotations

from typing import TYPE_CHECKING

import mlprosection.functions as F
from .base import Layer

if TYPE_CHECKING:
    from mlprosection import Tensor

class Activation(Layer):
    pass


class Relu(Activation):
    def __init__(self):
        self.mask: Tensor = None

    def forward_manual(self, x: Tensor):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward_manual(self, dout: Tensor):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid(Activation):
    def __init__(self):
        self.out: Tensor = None

    def forward_manual(self, x: Tensor):
        out = F.sigmoid(x)
        self.out = out
        return out

    def backward_manual(self, dout: Tensor):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Softmax(Activation):
    def __init__(self):
        self.out: Tensor = None

    def forward_manual(self, x: Tensor):
        self.out = F.softmax(x)
        return self.out

    def backward_manual(self, dout: Tensor):
        dx = self.out * dout
        sumdx = dx.sum(axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx

