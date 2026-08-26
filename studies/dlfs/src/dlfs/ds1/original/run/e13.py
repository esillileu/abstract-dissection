"""Original ch05 two-layer network backpropagation training."""

from __future__ import annotations

from pathlib import Path

from dlfs.original_runtime.measurement import OriginalMeasurements
from dlfs.original_runtime.runtime_context import budget, master_seed

from .common import (
    COMMON_SOURCES,
    Trial,
    importlib,
    np,
    save_csv,
    save_params,
    source_imports,
)

SOURCE = "ch05/train_neuralnet.py"


def _run(worktree: Path, output: Path) -> None:
    with source_imports(worktree):
        load_mnist = importlib.import_module("dataset.mnist").load_mnist
        network_cls = importlib.import_module("ch05.two_layer_net").TwoLayerNet
        np.random.seed(master_seed())
        (x_train, t_train), (x_test, t_test) = load_mnist(
            normalize=True, one_hot_label=True
        )
        network = network_cls(input_size=784, hidden_size=50, output_size=10)
        iterations = budget("max_updates", 10000)
        batch_size = 100
        learning_rate = 0.1
        train_size = x_train.shape[0]
        iter_per_epoch = max(train_size / batch_size, 1)
        rows = []
        measurements = OriginalMeasurements(output)
        with measurements.training():
            for index in range(iterations):
                batch_mask = np.random.choice(train_size, batch_size)
                x_batch, t_batch = x_train[batch_mask], t_train[batch_mask]
                gradient = network.gradient(x_batch, t_batch)
                for key in ("W1", "b1", "W2", "b2"):
                    network.params[key] -= learning_rate * gradient[key]
                if index % iter_per_epoch == 0:
                    rows.extend(
                        [
                            {
                                "update": index + 1,
                                "epoch": int(index / iter_per_epoch) + 1,
                                "split": "train",
                                "accuracy": network.accuracy(x_train, t_train),
                            },
                            {
                                "update": index + 1,
                                "epoch": int(index / iter_per_epoch) + 1,
                                "split": "test",
                                "accuracy": network.accuracy(x_test, t_test),
                            },
                        ]
                    )
        measurements.save(network.params)
        save_params(output / "checkpoint.npz", network.params)
    save_csv(output / "metrics.csv", rows)


TRIALS = (
    Trial(
        "dlfs1.ch05.two-layer-net.backprop",
        "numpy",
        {
            "gradient_method": "backprop",
            "updates": 10000,
            "batch_size": 100,
            "learning_rate": 0.1,
        },
        (*COMMON_SOURCES, SOURCE, "ch05/two_layer_net.py"),
        _run,
    ),
)
