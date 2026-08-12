"""Linear layers shared by ordinary and time-distributed models."""

from __future__ import annotations

from mlprosection import Tensor
from mlprosection.core.backend import Backend, resolve_backend

from .base import Layer
from ..initailizer import Initializer
from ..types import Parameter


class MatMul(Layer):
    def __init__(self, in_features: int, out_features: int, weight_init: Initializer | None = None, backend: Backend | None = None) -> None:
        super().__init__(resolve_backend(backend) if backend is not None else None)
        rng = self._backend.random_stream("model_init")
        self.W = Parameter(0.01 * rng.randn(in_features, out_features).astype(self._backend.float_dtype), backend=self._backend, name="W")
        if weight_init:
            self.W = weight_init(self.W)
        self.x: Tensor | None = None

    def forward_manual(self, x: Tensor) -> Tensor:
        self.x = x
        return Tensor(x.data @ self.W.data, backend=x.backend)

    def backward_manual(self, dout: Tensor) -> Tensor:
        if self.x is None:
            raise RuntimeError("forward() must be called before backward()")
        self.W.grad[...] = self.x.data.T @ dout.data
        return Tensor(dout.data @ self.W.data.T, backend=dout.backend)

    def reset_weight(self, weight_init: Initializer) -> None:
        self.W = weight_init(self.W)


class Affine(MatMul):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_init: Initializer | None = None,
        bias_init: Initializer | None = None,
        backend: Backend | None = None,
        *,
        weight: Parameter | None = None,
        transpose_weight: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, weight_init, backend or (weight.backend if weight is not None else None))
        if weight is not None:
            self.W = weight
        self.transpose_weight = transpose_weight
        self.b = Parameter(self._backend.xp.zeros(out_features, dtype=self._backend.float_dtype), backend=self._backend, name="b")
        if bias_init:
            self.b = bias_init(self.b)

    def _weight_data(self):
        return self.W.data.T if self.transpose_weight else self.W.data

    def forward_manual(self, x: Tensor) -> Tensor:
        self.x = x
        return Tensor(x.data @ self._weight_data() + self.b.data, backend=x.backend)

    def backward_manual(self, dout: Tensor) -> Tensor:
        if self.x is None:
            raise RuntimeError("forward() must be called before backward()")
        gradient = self.x.data.T @ dout.data
        self.W.grad[...] = gradient.T if self.transpose_weight else gradient
        self.b.grad[...] = dout.data.sum(axis=0)
        return Tensor(dout.data @ self._weight_data().T, backend=dout.backend)

    def reset_bias(self, bias_init: Initializer) -> None:
        self.b = bias_init(self.b)
