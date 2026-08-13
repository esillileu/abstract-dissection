"""Original ch06 weight-decay overfitting run."""

from __future__ import annotations

from pathlib import Path

from exp.deepscratch.original_runtime.measurement import OriginalMeasurements
from exp.deepscratch.original_runtime.runtime_context import budget, master_seed

from .common import COMMON_SOURCES, Trial, importlib, patch_cupy_modules, save_csv, save_params, source_imports


def run(worktree: Path, output: Path, *, weight_decay_lambda: float) -> None:
    rows = []
    with source_imports(worktree):
        load_mnist = importlib.import_module("dataset.mnist").load_mnist
        network_cls = importlib.import_module("common.multi_layer_net").MultiLayerNet
        optimizer = importlib.import_module("common.optimizer").SGD(lr=0.01)
        (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
        x_train, t_train = x_train[:300], t_train[:300]
        xp, _ = patch_cupy_modules(worktree, tuple(f"common.{name}" for name in ("functions", "gradient", "layers", "multi_layer_net", "optimizer")))
        x_train, t_train = xp.asarray(x_train), xp.asarray(t_train)
        x_test, t_test = xp.asarray(x_test), xp.asarray(t_test)
        xp.random.seed(master_seed())
        network = network_cls(
            784, [100] * 6, 10, weight_decay_lambda=weight_decay_lambda
        )
        measurements = OriginalMeasurements(output)
        with measurements.training():
            for epoch in range(budget("max_epochs", 201)):
                for _ in range(1 if epoch == 0 else 3):
                    mask = xp.random.choice(300, 100)
                    x_batch, t_batch = x_train[mask], t_train[mask]
                    optimizer.update(
                        network.params, network.gradient(x_batch, t_batch)
                    )
                for split, x_value, t_value, count in (
                    ("train", x_train, t_train, 300),
                    ("test", x_test, t_test, len(x_test)),
                ):
                    rows.append(
                        {
                            "update": epoch * 3 + 1,
                            "epoch": epoch,
                            "split": split,
                            "evaluation_set_id": f"mnist-{split}-full",
                            "example_count": count,
                            "accuracy": float(network.accuracy(x_value, t_value)),
                        }
                    )
        measurements.save(network.params, scope="source_training_and_evaluation")
        save_params(output / "checkpoint.npz", network.params)
    save_csv(output / "metrics.csv", rows)


TRIALS = (
    Trial(
        "dlfs1.ch06.weight-decay.off",
        "numpy",
        {"weight_decay_lambda": 0.0, "epochs": 201, "train_size": 300},
        COMMON_SOURCES + ("ch06/overfit_weight_decay.py",),
        lambda worktree, output: run(
            worktree, output, weight_decay_lambda=0.0
        ),
    ),
    Trial(
        "dlfs1.ch06.weight-decay.lambda-01",
        "numpy",
        {"weight_decay_lambda": 0.1, "epochs": 201, "train_size": 300},
        COMMON_SOURCES + ("ch06/overfit_weight_decay.py",),
        lambda worktree, output: run(
            worktree, output, weight_decay_lambda=0.1
        ),
    ),
)
