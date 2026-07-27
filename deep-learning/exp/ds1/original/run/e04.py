"""Original ch06 dropout overfitting run."""

from __future__ import annotations

from pathlib import Path

from .common import COMMON_SOURCES, Trial, importlib, np, save_csv, source_imports


def run(worktree: Path, output: Path) -> None:
    with source_imports(worktree):
        load_mnist = importlib.import_module("dataset.mnist").load_mnist
        network_cls = importlib.import_module(
            "common.multi_layer_net_extend"
        ).MultiLayerNetExtend
        trainer_cls = importlib.import_module("common.trainer").Trainer
        (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
        x_train, t_train = x_train[:300], t_train[:300]
        np.random.seed(1)
        network = network_cls(
            784,
            [100] * 6,
            10,
            use_dropout=True,
            dropout_ration=0.2,
        )
        trainer = trainer_cls(
            network,
            x_train,
            t_train,
            x_test,
            t_test,
            epochs=301,
            mini_batch_size=100,
            optimizer="sgd",
            optimizer_param={"lr": 0.01},
            verbose=False,
        )
        trainer.train()
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
        "dlfs1.ch06.dropout.on-ratio-02",
        "numpy",
        {"use_dropout": True, "dropout_ratio": 0.2, "epochs": 301},
        COMMON_SOURCES + ("ch06/overfit_dropout.py",),
        run,
    ),
)
