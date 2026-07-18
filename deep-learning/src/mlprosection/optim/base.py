from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, TYPE_CHECKING


if TYPE_CHECKING:
    from mlprosection.nn.types import Parameter
    from .transform import OptimizerTransform
    from ..nn.types import NamedParameters


class Optimizer(ABC):
    def __init__(
        self,
        params: NamedParameters,
        *,
        pre_step_hooks: Iterable[OptimizerTransform] | None = None,
        post_step_hooks: Iterable[OptimizerTransform] | None = None,
    ) -> None:
        self.params = list(params)

        self.pre_step_hooks = list(pre_step_hooks or ())
        self.post_step_hooks = list(post_step_hooks or ())

    def update(self) -> None:
        active_params = [
            (name, param) for name, param in self.params if param.grad is not None
        ]

        if not active_params:
            return

        self.before_step()

        for hook in self.pre_step_hooks:
            hook(active_params)

        for name, param in active_params:
            self.update_one(name, param)

        for hook in self.post_step_hooks:
            hook(active_params)

        self.after_step()

    @abstractmethod
    def update_one(
        self,
        name: str,
        param: Parameter,
    ) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for _, param in self.params:
            if param.grad is not None:
                param.grad[...] = 0

    def before_step(self) -> None:
        pass

    def after_step(self) -> None:
        pass
