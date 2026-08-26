"""Original ch06 activation-distribution observation."""

from __future__ import annotations

from pathlib import Path

from dlfs.original_runtime.runtime_context import array_module, master_seed

from .common import Trial, save_npz


def run(_worktree: Path, output: Path) -> None:
    xp = array_module()
    xp.random.seed(master_seed())
    x = xp.random.randn(1000, 100)
    activations = {}
    for index in range(5):
        weights = xp.random.randn(100, 100)
        x = 1 / (1 + xp.exp(-xp.dot(x, weights)))
        activations[f"layer_{index + 1}"] = x
    save_npz(output / "activations.npz", **activations)


TRIALS = (
    Trial(
        "dlfs1.ch06.activation.sigmoid-std-1",
        "numpy",
        {
            "activation": "sigmoid",
            "weight_scale": 1.0,
            "examples": 1000,
            "nodes": 100,
            "layers": 5,
        },
        ("ch06/weight_init_activation_histogram.py",),
        run,
    ),
)
