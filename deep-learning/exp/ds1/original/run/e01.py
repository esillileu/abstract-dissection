"""Original ch06 MNIST optimizer comparison."""

from __future__ import annotations

from pathlib import Path

from exp.original.measurement import OriginalMeasurements
from exp.original.runtime_context import budget, master_seed

from .common import COMMON_SOURCES, Trial, importlib, patch_cupy_modules, rows_for_series, save_csv, save_params, source_imports


SOURCE = "ch06/optimizer_compare_mnist.py"
CONDITIONS = ("SGD", "Momentum", "AdaGrad", "Adam")


def _run(condition: str, worktree: Path, output: Path) -> None:
    with source_imports(worktree):
        load_mnist = importlib.import_module("dataset.mnist").load_mnist
        network_cls = importlib.import_module("common.multi_layer_net").MultiLayerNet
        optim = importlib.import_module("common.optimizer")
        (x_train, t_train), _ = load_mnist(normalize=True)
        xp, _ = patch_cupy_modules(worktree, tuple(f"common.{name}" for name in ("functions", "gradient", "layers", "multi_layer_net", "optimizer")))
        x_train, t_train = xp.asarray(x_train), xp.asarray(t_train)
        xp.random.seed(master_seed())
        networks = {
            key: network_cls(784, [100, 100, 100, 100], 10)
            for key in CONDITIONS
        }
        optimizers = {
            "SGD": optim.SGD(),
            "Momentum": optim.Momentum(),
            "AdaGrad": optim.AdaGrad(),
            "Adam": optim.Adam(),
        }
        losses = []
        network = networks[condition]
        optimizer = optimizers[condition]
        measurements = OriginalMeasurements(output)
        with measurements.training():
            for _ in range(budget("max_updates", 2000)):
                mask = xp.random.choice(len(x_train), 128)
                x_batch, t_batch = x_train[mask], t_train[mask]
                optimizer.update(
                    network.params,
                    network.gradient(x_batch, t_batch),
                )
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
        f"dlfs1.ch06.optimizer-mnist.{name.lower()}",
        "numpy",
        {"condition": name, "updates": 2000, "batch_size": 128},
        COMMON_SOURCES + (SOURCE,),
        lambda worktree, output, name=name: _run(name, worktree, output),
    )
    for name in CONDITIONS
)
