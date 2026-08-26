"""Objective translation adapter for DS1."""

from __future__ import annotations

from typing import Any

from deepscratch.nn.objective import SoftmaxCrossEntropy


def build_ds1_objective(config: dict[str, object], backend: Any) -> SoftmaxCrossEntropy:
    """Instantiate a DeepScratch objective from a DS1 objective configuration dictionary."""
    name = str(config.get("name", "SoftmaxCrossEntropy"))
    if name == "SoftmaxCrossEntropy":
        return SoftmaxCrossEntropy(
            reduction=str(config.get("reduction", "mean")),
            backend=backend,
        )
    raise ValueError(f"unknown objective name: {name}")
