"""Model translation and parameter extraction adapters for DS1."""

from __future__ import annotations

from typing import Any

import numpy as np
from deepscratch.nn.model.architecture import MLP, DeepCNN, SimpleCNN, TwoLayerNet


def build_ds1_model(config: dict[str, object], *, dropout_rng=None) -> Any:
    """Instantiate a DeepScratch model from a DS1 model configuration dictionary."""
    name = str(config.get("name", "MLP"))
    values = {
        key: value
        for key, value in config.items()
        if key
        not in {
            "name",
            "family",
            "task_type",
            "input_shape",
            "output_shape",
            "structure_signature",
            "use_batchnorm",
            "use_dropout",
            "num_hidden_layers",
            "num_conv_layers",
            "normalization",
            "model/flops",
            "model/macs",
        }
    }
    if name == "TwoLayerNet":
        values.pop("gradient_method", None)
        values.pop("numerical_step", None)
        return TwoLayerNet(
            **values, numerical_step=float(config.get("numerical_step", 1e-4))
        )
    if name == "MLP":
        if "activation" in values:
            values["activation_name"] = values.pop("activation")
        if "use_batchnorm" in config:
            values["batchnorm"] = bool(config["use_batchnorm"])
    else:
        values.pop("activation", None)
    if name == "MLP":
        return MLP(**values, dropout_rng=dropout_rng)
    if name == "SimpleCNN":
        return SimpleCNN(**values)
    if name == "DeepCNN":
        return DeepCNN(**values, dropout_rng=dropout_rng)
    raise ValueError(f"unknown model name: {name}")


def training_parameters(model: Any, objective: Any) -> list[tuple[str, Any]]:
    """Extract named parameter pairs for optimizer construction."""
    return [
        *((f"model.{name}", parameter) for name, parameter in model.named_parameters()),
        *(
            (f"objective.{name}", parameter)
            for name, parameter in objective.named_parameters()
        ),
    ]


def book_gradients(model: Any) -> dict[str, np.ndarray]:
    """Extract standard DS1 TwoLayerNet gradients as host numpy arrays."""
    named = dict(model.named_parameters())
    return {
        book_name: parameter.backend.to_numpy(parameter.grad).copy()
        for book_name, parameter_name in (
            ("W1", "layers.0.W"),
            ("b1", "layers.0.b"),
            ("W2", "layers.2.W"),
            ("b2", "layers.2.b"),
        )
        for parameter in (named[parameter_name],)
    }


def initializer_scale(initializer: str, fan_in: int) -> float:
    """Compute Gaussian standard deviation scaling for layer initialization."""
    if initializer == "he":
        return float(np.sqrt(2.0 / fan_in))
    if initializer == "xavier":
        return float(np.sqrt(1.0 / fan_in))
    if initializer.startswith("std:"):
        return float(initializer.split(":", 1)[1])
    return 1.0


def activation_fn(value: np.ndarray, name: str) -> np.ndarray:
    """Apply activation function to host numpy array."""
    if name == "relu":
        return np.maximum(0.0, value)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-value))
    if name == "tanh":
        return np.tanh(value)
    raise ValueError(f"unknown activation: {name}")
