"""Original ch06 weight-initialization comparison."""

from __future__ import annotations

from pathlib import Path

from exp.deepscratch.original_runtime.measurement import OriginalMeasurements
from exp.deepscratch.original_runtime.runtime_context import budget, master_seed

from .common import COMMON_SOURCES, Trial, importlib, patch_cupy_modules, rows_for_series, save_csv, save_params, source_imports


SOURCE = "ch06/weight_init_compare.py"
INITIALIZERS = {"std=0.01": 0.01, "Xavier": "sigmoid", "He": "relu"}


def _run(condition: str, worktree: Path, output: Path) -> None:
    with source_imports(worktree):
        load_mnist = importlib.import_module("dataset.mnist").load_mnist
        network_cls = importlib.import_module("common.multi_layer_net").MultiLayerNet
        sgd = importlib.import_module("common.optimizer").SGD(lr=0.01)
        (x_train, t_train), _ = load_mnist(normalize=True)
        xp, _ = patch_cupy_modules(worktree, tuple(f"common.{name}" for name in ("functions", "gradient", "layers", "multi_layer_net", "optimizer")))
        x_train, t_train = xp.asarray(x_train), xp.asarray(t_train)
        xp.random.seed(master_seed())
        networks = {
            name: network_cls(
                784, [100, 100, 100, 100], 10, weight_init_std=value
            )
            for name, value in INITIALIZERS.items()
        }
        losses = []
        network = networks[condition]
        measurements = OriginalMeasurements(output)
        with measurements.training():
            for _ in range(budget("max_updates", 2000)):
                mask = xp.random.choice(len(x_train), 128)
                x_batch, t_batch = x_train[mask], t_train[mask]
                sgd.update(network.params, network.gradient(x_batch, t_batch))
                losses.append(network.loss(x_batch, t_batch))
        measurements.save(network.params)
        save_params(output / "checkpoint.npz", network.params)
    save_csv(
        output / "metrics.csv",
        rows_for_series(
            losses, condition=condition, metric="train/objective", batch_size=128
        ),
    )


TRIALS = tuple(
    Trial(
        "dlfs1.ch06.init-compare."
        + {"std=0.01": "std-001", "Xavier": "xavier", "He": "he"}[name],
        "numpy",
        {"initializer": name, "updates": 2000, "batch_size": 128},
        COMMON_SOURCES + (SOURCE,),
        lambda worktree, output, name=name: _run(name, worktree, output),
    )
    for name in INITIALIZERS
)
