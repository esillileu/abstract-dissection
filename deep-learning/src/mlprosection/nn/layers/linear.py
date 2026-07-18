from .base import Layer
from ..types import Parameter
from ..initailizer import Initializer
from mlprosection.core.backend import Backend


class MatMul(Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_init: Initializer | None = None,
        backend: Backend | None = None,
    ):
        super().__init__(backend)
        self.W = Parameter(
            0.01
            * self._backend.xp.random.randn(
                in_features,
                out_features,
            ).astype(self._backend.float_dtype),
            backend=backend,
            name="W",
        )

        if weight_init:
            self.W = weight_init(self.W)

        self.x = None

        return backend

    def forward_manual(self, x):
        self.x = x
        return x @ self.W

    def backward_manual(self, dout):
        assert self.x is not None
        self.W.set_grad(self.x.T @ dout)
        return dout @ self.W.T

    def reset_weight(self, weight_init: Initializer):
        self.W = weight_init(self.W)


class Affine(MatMul):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_init: Initializer | None = None,
        bias_init: Initializer | None = None,
        backend: Backend = None,
    ):
        super().__init__(in_features, out_features, weight_init, backend)

        self.b = Parameter(
            self._backend.xp.zeros(out_features, dtype=self._backend.float_dtype),
            backend=backend,
            name="b",
        )
        if bias_init:
            self.b = bias_init(self.b)
        self.x = None

    def forward_manual(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward_manual(self, dout):
        assert self.x is not None
        self.W.set_grad(self.x.T @ dout)
        self.b.set_grad(dout.sum(axis=0))
        return dout @ self.W.T

    def forward_auto(self):
        pass

    def reset_bias(self, bias_init: Initializer):
        self.b = bias_init(self.b)
