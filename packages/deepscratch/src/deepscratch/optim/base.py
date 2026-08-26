from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepscratch.nn.types import Parameter

    from ..nn.types import NamedParameters
    from .transform import OptimizerTransform


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

    def state_dict(self) -> dict[str, object]:
        """Portable optimizer state for checkpointing at epoch boundaries."""
        values: dict[str, object] = {"type": type(self).__name__}
        for name, value in self.__dict__.items():
            if name in {"params", "pre_step_hooks", "post_step_hooks"}:
                continue
            if isinstance(value, (int, float, str, bool)):
                values[name] = value
            elif isinstance(value, dict):
                values[name] = {
                    key: parameter.backend.to_numpy(parameter_value).copy()
                    for key, parameter_value in value.items()
                    if hasattr(parameter_value, "shape")
                    for parameter in [dict(self.params)[key]]
                }
        return values

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("type") != type(self).__name__:
            raise ValueError(
                f"checkpoint optimizer {state.get('type')} does not match {type(self).__name__}"
            )
        named_params = dict(self.params)
        for name, value in state.items():
            if name in {"type", "params", "pre_step_hooks", "post_step_hooks"}:
                continue
            if isinstance(value, dict) and hasattr(self, name):
                target = getattr(self, name)
                if isinstance(target, dict):
                    for key, array in value.items():
                        if key in target and key in named_params:
                            target[key][...] = named_params[key].backend.xp.asarray(
                                array
                            )
            elif isinstance(value, (int, float, str, bool)):
                setattr(self, name, value)
