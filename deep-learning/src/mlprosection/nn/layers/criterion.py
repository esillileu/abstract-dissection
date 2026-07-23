from __future__ import annotations

from typing import TYPE_CHECKING

import mlprosection.functions as F
from mlprosection import Tensor
from .base import Layer

if TYPE_CHECKING:
    pass

class Criterion(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.y: Tensor | None = None
        self.t: Tensor | None = None

    def check_forward(self) -> None:
        if self.y is None or self.t is None:
            raise RuntimeError("forward() must be called before backward()")

    def backward_manual(self, dout: Tensor | None = None) -> Tensor:
        raise NotImplementedError


class SoftmaxWithLoss(Criterion):
    def __init__(self) -> None:
        super().__init__()

    def forward_manual(self, x: Tensor, t: Tensor) -> Tensor:
        self.y = F.softmax(x)
        self.t = t

        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        return F.cee(self.y, self.t)

    def backward_manual(self, dout: Tensor | None = None) -> Tensor:
        super().check_forward()

        if dout is None:
            dout = Tensor(1, backend=self.y.backend)

        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[Tensor.arange(batch_size, backend=self.y.backend), self.t] -= 1
        dx *= dout
        dx /= batch_size
        return dx


class SigmoidWithLoss(Criterion):
    def __init__(self):
        super().__init__()

    def forward_manual(self, x: Tensor, t: Tensor) -> Tensor:
        self.y = F.sigmoid(x)
        self.t = t

        self.loss = F.cee(self.t.backend.xp.c_[1 - self.y, self.y], self.t)
        return self.loss

    def backward_manual(self, dout: Tensor | None = None) -> Tensor:
        super().check_forward()

        if dout is None:
            dout = Tensor(1, backend=self.y.backend)

        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx
