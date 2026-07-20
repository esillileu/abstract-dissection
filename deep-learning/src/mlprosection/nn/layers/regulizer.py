from __future__ import annotations

from .base import Layer, Parameter

from mlprosection import Tensor


class Dropout(Layer):
    def __init__(self, dropout_ratio=0.5):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward_manual(self, x: Tensor):
        if self.training:
            self.mask = x.backend.xp.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward_manual(self, dout: Tensor):
        return dout * self.mask


class BatchNormalization(Layer):
    def __init__(
        self,
        gamma: float = 1.0,
        beta: float = 0.0,
        momentum: float = 0.9,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must satisfy 0 <= momentum < 1")

        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.gamma_init = gamma
        self.beta_init = beta

        self.gamma: Parameter | None = None
        self.beta: Parameter | None = None

        self.momentum = momentum
        self.eps = eps

        self.input_shape: tuple[int, ...] | None = None

        self.running_mean = None
        self.running_var = None

        self.batch_size: int | None = None
        self.xc: Tensor | None = None
        self.xn: Tensor | None = None
        self.std: Tensor | None = None

    def _initialize(
        self,
        x: Tensor,
    ) -> None:
        feature_size = x.shape[1]
        xp = x.backend.xp

        if self.gamma is None:
            self.gamma = Parameter(
                Tensor(
                    xp.full(
                        feature_size,
                        self.gamma_init,
                        dtype=x.dtype,
                    ),
                    backend=x.backend,
                )
            )

        if self.beta is None:
            self.beta = Parameter(
                Tensor(
                    xp.full(
                        feature_size,
                        self.beta_init,
                        dtype=x.dtype,
                    ),
                    backend=x.backend,
                )
            )

        if self.running_mean is None:
            self.running_mean = xp.zeros(feature_size, dtype=x.dtype)

        if self.running_var is None:
            self.running_var = xp.zeros(feature_size, dtype=x.dtype)

    def forward_manual(
        self,
        x: Tensor,
    ) -> Tensor:
        self.input_shape = x.shape

        if x.ndim != 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)

        self._initialize(x)

        out = self._forward_2d(
            x,
            train_flg=self.training,
        )

        return out.reshape(*self.input_shape)

    def _forward_2d(
        self,
        x: Tensor,
        *,
        train_flg: bool,
    ) -> Tensor:
        assert self.gamma is not None
        assert self.beta is not None
        assert self.running_mean is not None
        assert self.running_var is not None

        if train_flg:
            mean = x.mean(axis=0)
            xc = x - mean

            var = (xc * xc).mean(axis=0)
            std = (var + self.eps).sqrt()
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std

            self.running_mean[...] = (
                self.momentum * self.running_mean + (1.0 - self.momentum) * mean.data
            )

            self.running_var[...] = (
                self.momentum * self.running_var + (1.0 - self.momentum) * var.data
            )

        else:
            running_mean = Tensor(
                self.running_mean,
                backend=x.backend,
                requires_grad=False,
            )

            running_var = Tensor(
                self.running_var,
                backend=x.backend,
                requires_grad=False,
            )

            xc = x - running_mean
            xn = xc / (running_var + self.eps).sqrt()

        return self.gamma * xn + self.beta

    def backward_manual(
        self,
        dout: Tensor,
    ) -> Tensor:
        if self.input_shape is None:
            raise RuntimeError("forward must be called before backward")

        if dout.ndim != 2:
            batch_size = dout.shape[0]
            dout = dout.reshape(batch_size, -1)

        dx = self._backward_2d(dout)

        return dx.reshape(*self.input_shape)

    def _backward_2d(
        self,
        dout: Tensor,
    ) -> Tensor:
        if (
            self.batch_size is None
            or self.xc is None
            or self.xn is None
            or self.std is None
        ):
            raise RuntimeError("training forward must be called before backward")

        assert self.gamma is not None
        assert self.beta is not None

        dbeta = dout.sum(axis=0)
        dgamma = (self.xn * dout).sum(axis=0)

        dxn = self.gamma * dout
        dxc = dxn / self.std

        std_sq = self.std * self.std

        dstd = -((dxn * self.xc) / std_sq).sum(axis=0)

        dvar = 0.5 * dstd / self.std

        dxc = dxc + (2.0 / self.batch_size) * self.xc * dvar

        dmu = dxc.sum(axis=0)
        dx = dxc - dmu / self.batch_size

        self.gamma.grad = dgamma.data
        self.beta.grad = dbeta.data

        return dx
