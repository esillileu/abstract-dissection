"""Dataset loader adapter for DS1 with runtime path injection."""

from __future__ import annotations

from typing import Any

from deepscratch.datasets import load_mnist

from repro_core.context.paths import RuntimePaths


def load_ds1_mnist(
    *,
    flatten: bool = True,
    one_hot_label: bool = False,
    gpu: bool = False,
) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    """Load MNIST with canonical repository path resolution."""
    mnist_dir = RuntimePaths.from_environment().dataset("mnist")
    return load_mnist(
        flatten=flatten,
        one_hot_label=one_hot_label,
        gpu=gpu,
        data_dir=mnist_dir,
    )
