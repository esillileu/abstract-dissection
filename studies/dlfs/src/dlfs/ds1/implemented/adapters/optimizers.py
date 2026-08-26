"""Optimizer translation adapter for DS1."""

from __future__ import annotations

from typing import Any

from deepscratch.optim.SGD import SGD, AdaGrad, Adam, Momentum
from deepscratch.optim.transform import L2Regularization


def build_ds1_optimizer(config: dict[str, object], params: Any) -> Any:
    """Instantiate a DeepScratch optimizer from a DS1 optimizer configuration dictionary."""
    name = str(config.get("name", "sgd"))
    learning_rate = float(config.get("learning_rate", 0.01))
    weight_decay = float(config.get("weight_decay", 0.0))
    hooks = [L2Regularization(weight_decay)] if weight_decay else None
    if name == "sgd":
        return SGD(params, lr=learning_rate, pre_step_hooks=hooks)
    if name == "momentum":
        return Momentum(
            params,
            lr=learning_rate,
            momentum=float(config.get("momentum", 0.9)),
            pre_step_hooks=hooks,
        )
    if name == "adagrad":
        return AdaGrad(
            params,
            lr=learning_rate,
            eps=float(config.get("eps", 1e-7)),
            pre_step_hooks=hooks,
        )
    if name == "adam":
        return Adam(
            params,
            lr=learning_rate,
            beta1=float(config.get("beta1", 0.9)),
            beta2=float(config.get("beta2", 0.999)),
            eps=float(config.get("eps", 1e-7)),
            pre_step_hooks=hooks,
        )
    raise ValueError(f"unknown optimizer: {name}")
