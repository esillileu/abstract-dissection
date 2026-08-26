"""Original ch06 dropout overfitting run."""

from __future__ import annotations

from pathlib import Path

from dlfs.original_runtime.measurement import OriginalMeasurements
from dlfs.original_runtime.runtime_context import budget, master_seed

from .common import (
    COMMON_SOURCES,
    Trial,
    importlib,
    patch_cupy_modules,
    save_csv,
    save_params,
    source_imports,
)


def run(worktree: Path, output: Path, *, use_dropout: bool) -> None:
    with source_imports(worktree):
        load_mnist = importlib.import_module("dataset.mnist").load_mnist
        network_cls = importlib.import_module(
            "common.multi_layer_net_extend"
        ).MultiLayerNetExtend
        trainer_cls = importlib.import_module("common.trainer").Trainer
        (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
        x_train, t_train = x_train[:300], t_train[:300]
        xp, _ = patch_cupy_modules(
            worktree,
            tuple(
                f"common.{name}"
                for name in (
                    "functions",
                    "gradient",
                    "layers",
                    "multi_layer_net_extend",
                    "optimizer",
                    "trainer",
                )
            ),
        )
        x_train, t_train = xp.asarray(x_train), xp.asarray(t_train)
        x_test, t_test = xp.asarray(x_test), xp.asarray(t_test)
        xp.random.seed(master_seed())
        network = network_cls(
            784,
            [100] * 6,
            10,
            use_dropout=use_dropout,
            dropout_ration=0.2,
        )
        trainer = trainer_cls(
            network,
            x_train,
            t_train,
            x_test,
            t_test,
            epochs=budget("max_epochs", 301),
            mini_batch_size=100,
            optimizer="sgd",
            optimizer_param={"lr": 0.01},
            verbose=False,
        )
        measurements = OriginalMeasurements(output)
        with measurements.training():
            trainer.train()
        measurements.save(network.params, scope="source_training_and_evaluation")
        save_params(output / "checkpoint.npz", network.params)
        rows = []
        for epoch, (train_acc, test_acc) in enumerate(
            zip(trainer.train_acc_list, trainer.test_acc_list, strict=True)
        ):
            rows.extend(
                (
                    {
                        "update": epoch * 3,
                        "epoch": epoch,
                        "split": "train",
                        "accuracy": float(train_acc),
                    },
                    {
                        "update": epoch * 3,
                        "epoch": epoch,
                        "split": "test",
                        "accuracy": float(test_acc),
                    },
                )
            )
    save_csv(output / "metrics.csv", rows)


TRIALS = (
    Trial(
        "dlfs1.ch06.dropout.off",
        "numpy",
        {"use_dropout": False, "dropout_ratio": 0.0, "epochs": 301},
        (*COMMON_SOURCES, "ch06/overfit_dropout.py"),
        lambda worktree, output: run(worktree, output, use_dropout=False),
    ),
    Trial(
        "dlfs1.ch06.dropout.on-ratio-02",
        "numpy",
        {"use_dropout": True, "dropout_ratio": 0.2, "epochs": 301},
        (*COMMON_SOURCES, "ch06/overfit_dropout.py"),
        lambda worktree, output: run(worktree, output, use_dropout=True),
    ),
)
