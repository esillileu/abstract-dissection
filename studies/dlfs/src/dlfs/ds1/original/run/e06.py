"""Original ch07 SimpleConvNet through a module-global CuPy adapter."""

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
    to_host,
)

GPU_MODULES = (
    "common.functions",
    "common.gradient",
    "common.layers",
    "common.optimizer",
    "common.trainer",
    "common.util",
    "ch07.simple_convnet",
)


def run(worktree: Path, output: Path) -> None:
    with source_imports(worktree):
        cp, modules = patch_cupy_modules(worktree, GPU_MODULES)
        load_mnist = importlib.import_module("dataset.mnist").load_mnist
        (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
        x_train, t_train = cp.asarray(x_train), cp.asarray(t_train)
        x_test, t_test = cp.asarray(x_test), cp.asarray(t_test)
        cp.random.seed(master_seed())
        network = modules["ch07.simple_convnet"].SimpleConvNet(
            input_dim=(1, 28, 28),
            conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
            hidden_size=100,
            output_size=10,
            weight_init_std=0.01,
        )
        initial_w1 = to_host(network.params["W1"])
        trainer = modules["common.trainer"].Trainer(
            network,
            x_train,
            t_train,
            x_test,
            t_test,
            epochs=budget("max_epochs", 20),
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
        save_params(
            output / "checkpoint.npz",
            network.params,
            initial_W1=initial_w1,
        )
    save_csv(output / "metrics.csv", rows)


TRIALS = (
    Trial(
        "dlfs1.ch07.simple-convnet",
        "cupy",
        {
            "epochs": 20,
            "batch_size": 100,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "evaluation_size": 1000,
        },
        (
            *COMMON_SOURCES,
            "ch07/simple_convnet.py",
            "ch07/train_convnet.py",
            "ch07/visualize_filter.py",
        ),
        run,
    ),
)
