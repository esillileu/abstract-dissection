"""Optimizer translation adapter for DS2."""

from __future__ import annotations

from typing import Any

from deepscratch.optim.SGD import SGD, Adam, SparseAdam
from deepscratch.optim.transform import ClipGradNorm


def build_sequence_optimizer(
    config: dict[str, object],
    model: Any,
    objective: Any,
) -> Any:
    """Instantiate a DeepScratch optimizer for sequence models."""
    values = config.get("optimizer", {})
    if not isinstance(values, dict):
        raise ValueError("optimizer must be a mapping")
    name = str(values.get("name", "adam"))
    params = [
        *(
            (f"model.{param_name}", parameter)
            for param_name, parameter in model.named_parameters()
        ),
        *(
            (f"objective.{param_name}", parameter)
            for param_name, parameter in objective.named_parameters()
        ),
    ]
    default_max_grad = {
        "language_modeling": 0.25,
        "seq2seq": 5.0,
    }.get(str(config.get("kind")))
    policy = config.get("policy", {})
    max_grad = None
    if isinstance(policy, dict):
        val = policy.get("max_grad", default_max_grad)
        max_grad = None if val is None else float(val)
    elif default_max_grad is not None:
        max_grad = float(default_max_grad)

    hooks = None if max_grad is None else [ClipGradNorm(max_grad)]
    if name == "adam":
        return Adam(
            params,
            lr=float(values.get("learning_rate", 0.001)),
            pre_step_hooks=hooks,
        )
    if name == "sparse_adam":
        if not hasattr(model, "sparse_parameter_rows"):
            raise ValueError("sparse_adam requires a sparse-row model")
        return SparseAdam(
            params,
            row_indices={
                "model.W_in": lambda: model.sparse_parameter_rows()["W_in"],
                "model.W_out": lambda: model.sparse_parameter_rows()["W_out"],
            },
            lr=float(values.get("learning_rate", 0.001)),
        )
    if name == "sgd":
        return SGD(
            params,
            lr=float(values.get("learning_rate", 1.0)),
            pre_step_hooks=hooks,
        )
    raise ValueError(f"unsupported sequence optimizer: {name}")
