"""Original ch06 two-dimensional optimizer trajectories."""

from __future__ import annotations

from pathlib import Path

from exp.original.runtime_context import budget

from .common import COMMON_SOURCES, Trial, importlib, save_csv, source_imports


OPTIMIZERS = {
    "sgd": ("SGD", {"lr": 0.95}),
    "momentum": ("Momentum", {"lr": 0.1}),
    "adagrad": ("AdaGrad", {"lr": 1.5}),
    "adam": ("Adam", {"lr": 0.3}),
}


def _run(name: str, worktree: Path, output: Path) -> None:
    with source_imports(worktree):
        optimizer_module = importlib.import_module("common.optimizer")
        class_name, kwargs = OPTIMIZERS[name]
        optimizer = getattr(optimizer_module, class_name)(**kwargs)
        params = {"x": -7.0, "y": 2.0}
        grads = {"x": 0.0, "y": 0.0}
        rows = []
        for update in range(budget("max_updates", 30)):
            x, y = params["x"], params["y"]
            dx, dy = x / 10.0, 2.0 * y
            rows.append(
                {
                    "update": update,
                    "x": x,
                    "y": y,
                    "objective": x**2 / 20.0 + y**2,
                    "gradient_x": dx,
                    "gradient_y": dy,
                }
            )
            grads["x"], grads["y"] = dx, dy
            optimizer.update(params, grads)
    save_csv(output / "trajectory.csv", rows)


TRIALS = tuple(
    Trial(
        f"dlfs1.ch06.optimizer-path.{name}",
        "numpy",
        {"optimizer": class_name, **kwargs, "updates": 30, "initial": [-7.0, 2.0]},
        COMMON_SOURCES + ("ch06/optimizer_compare_naive.py",),
        lambda worktree, output, name=name: _run(name, worktree, output),
    )
    for name, (class_name, kwargs) in OPTIMIZERS.items()
)
