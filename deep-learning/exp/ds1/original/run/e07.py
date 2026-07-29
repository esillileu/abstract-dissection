"""Original ch08 DeepConvNet through a module-global CuPy adapter."""

from __future__ import annotations

from pathlib import Path

from exp.original.measurement import OriginalMeasurements

from .common import (
    COMMON_SOURCES,
    Trial,
    importlib,
    patch_cupy_modules,
    save_csv,
    save_params,
    source_imports,
)


GPU_MODULES = (
    "common.functions",
    "common.gradient",
    "common.layers",
    "common.optimizer",
    "common.trainer",
    "common.util",
    "ch08.deep_convnet",
)


def run(worktree: Path, output: Path) -> None:
    with source_imports(worktree):
        cp, modules = patch_cupy_modules(worktree, GPU_MODULES)
        load_mnist = importlib.import_module("dataset.mnist").load_mnist
        (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
        x_train, t_train = cp.asarray(x_train), cp.asarray(t_train)
        x_test, t_test = cp.asarray(x_test), cp.asarray(t_test)
        cp.random.seed(1)
        network = modules["ch08.deep_convnet"].DeepConvNet()
        trainer = modules["common.trainer"].Trainer(
            network,
            x_train,
            t_train,
            x_test,
            t_test,
            epochs=20,
            mini_batch_size=100,
            optimizer="Adam",
            optimizer_param={"lr": 0.001},
            evaluate_sample_num_per_epoch=1000,
            verbose=False,
        )
        measurements = OriginalMeasurements(output)
        with measurements.training():
            trainer.train()
        measurements.save(network.params, scope="source_training_and_evaluation")
        rows = []
        for epoch, (train_acc, test_acc) in enumerate(
            zip(trainer.train_acc_list, trainer.test_acc_list, strict=True)
        ):
            rows.extend(
                (
                    {"epoch": epoch, "split": "train", "accuracy": float(train_acc)},
                    {"epoch": epoch, "split": "test", "accuracy": float(test_acc)},
                )
            )
        rows.append(
            {
                "epoch": 20,
                "split": "test-full",
                "accuracy": float(network.accuracy(x_test, t_test)),
            }
        )
        save_params(output / "checkpoint.npz", network.params)
    save_csv(output / "metrics.csv", rows)


TRIALS = (
    Trial(
        "dlfs1.ch08.deep-convnet",
        "cupy",
        {
            "epochs": 20,
            "batch_size": 100,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "evaluation_size": 1000,
        },
        COMMON_SOURCES + ("ch08/deep_convnet.py", "ch08/train_deepnet.py"),
        run,
    ),
)
