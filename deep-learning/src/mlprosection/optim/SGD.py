from __future__ import annotations
from typing import Iterable, Dict, TYPE_CHECKING

from mlprosection.nn.types import Parameter
from .base import Optimizer

if TYPE_CHECKING:
    from mlprosection.core.backend.types import Array
    from mlprosection.nn.types import NamedParameters
    from .transform import OptimizerTransform


class SGD(Optimizer):
    def __init__(
        self,
        named_params: NamedParameters,
        lr: float = 0.01,
        *,
        pre_step_hooks: Iterable[OptimizerTransform] | None = None,
        post_step_hooks: Iterable[OptimizerTransform] | None = None,
    ):
        super().__init__(
            named_params,
            pre_step_hooks=pre_step_hooks,
            post_step_hooks=post_step_hooks,
        )
        self.lr: float = lr

    def update_one(self, _name: str, param: Parameter):
        param.data -= self.lr * param.grad


class Momentum(SGD):
    def __init__(
        self,
        named_params: NamedParameters,
        lr: float = 0.01,
        momentum: float = 0.9,
        *,
        pre_step_hooks: Iterable[OptimizerTransform] | None = None,
        post_step_hooks: Iterable[OptimizerTransform] | None = None,
    ) -> None:
        super().__init__(
            named_params,
            lr=lr,
            pre_step_hooks=pre_step_hooks,
            post_step_hooks=post_step_hooks,
        )
        self.m = momentum
        self.v: Dict[str, Array] = zero_arrays_like_named_params(self.params)

    def update_one(self, name: str, param: Parameter) -> None:
        v = self.v[name]
        v *= self.m
        v -= self.lr * param.grad
        param.data += v


class Nesterov(Momentum):
    def __init__(
        self,
        named_params: NamedParameters,
        lr: float = 0.01,
        momentum: float = 0.9,
        *,
        pre_step_hooks: Iterable[OptimizerTransform] | None = None,
        post_step_hooks: Iterable[OptimizerTransform] | None = None,
    ):
        super().__init__(
            named_params,
            lr,
            momentum,
            pre_step_hooks=pre_step_hooks,
            post_step_hooks=post_step_hooks,
        )

    def update_one(self, name, param):
        v = self.v[name]
        v *= self.m
        v -= self.lr * param.grad
        param.data += self.m * self.m * v
        param.data -= (1 + self.m) * self.lr * param.grad


class AdaGrad(SGD):
    def __init__(
        self,
        named_params: NamedParameters,
        lr: float = 0.01,
        *,
        eps=1e-7,
        pre_step_hooks: Iterable[OptimizerTransform] | None = None,
        post_step_hooks: Iterable[OptimizerTransform] | None = None,
    ):
        super().__init__(
            named_params,
            lr,
            pre_step_hooks=pre_step_hooks,
            post_step_hooks=post_step_hooks,
        )
        self.h: Dict[str, Array] = zero_arrays_like_named_params(self.params)
        self.eps = eps

    def update_one(self, name, param):
        xp = param.backend.xp

        h = self.h[name]
        h += param.grad * param.grad
        param.data -= self.lr * param.grad / (xp.sqrt(h) + self.eps)


class RMSprop(AdaGrad):
    def __init__(
        self,
        named_params: NamedParameters,
        lr: float = 0.01,
        decay_rate: float = 0.99,
        pre_step_hooks: Iterable[OptimizerTransform] | None = None,
        post_step_hooks: Iterable[OptimizerTransform] | None = None,
    ):
        super().__init__(
            named_params,
            lr,
            pre_step_hooks=pre_step_hooks,
            post_step_hooks=post_step_hooks,
        )
        self.decay_rate: float = decay_rate

    def update_one(self, name, param):
        xp = param.backend.xp

        h = self.h[name]
        h *= self.decay_rate
        h += (1 - self.decay_rate) * param.grad * param.grad
        param.data -= self.lr * param.grad / (xp.sqrt(h) + self.eps)


class Adam(SGD):
    def __init__(
        self,
        named_params: NamedParameters,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        *,
        pre_step_hooks: Iterable[OptimizerTransform] | None = None,
        post_step_hooks: Iterable[OptimizerTransform] | None = None,
        eps: float = 1e-7,
    ) -> None:
        super().__init__(
            named_params,
            lr=lr,
            pre_step_hooks=pre_step_hooks,
            post_step_hooks=post_step_hooks,
        )

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.iter = 0
        self.lr_t = lr

        self.m: Dict[str, Array] = zero_arrays_like_named_params(self.params)
        self.v: Dict[str, Array] = zero_arrays_like_named_params(self.params)

    def before_step(self) -> None:
        self.iter += 1

        self.lr_t = (
            self.lr
            * (1.0 - self.beta2**self.iter) ** 0.5
            / (1.0 - self.beta1**self.iter)
        )

    def update_one(self, name: str, param: Parameter) -> None:
        grad = param.grad
        m = self.m[name]
        v = self.v[name]
        xp = param.backend.xp

        m += (1.0 - self.beta1) * (grad - m)
        v += (1.0 - self.beta2) * (grad * grad - v)

        param.data -= self.lr_t * m / (xp.sqrt(v) + self.eps)


def zero_arrays_like_named_params(
    named_params: NamedParameters,
) -> Dict[str, Array]:
    return {
        name: param.backend.xp.zeros_like(param.data) for name, param in named_params
    }
