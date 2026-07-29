"""Original ch06 batch-normalization scale sweep."""

from __future__ import annotations

from pathlib import Path

from exp.original.measurement import OriginalMeasurements

from .common import COMMON_SOURCES, Trial, importlib, np, save_csv, source_imports


SCALES = np.logspace(0, -4, num=16)


def _run(scale_index: int, batchnorm: bool, worktree: Path, output: Path) -> None:
    with source_imports(worktree):
        load_mnist = importlib.import_module("dataset.mnist").load_mnist
        network_cls = importlib.import_module(
            "common.multi_layer_net_extend"
        ).MultiLayerNetExtend
        optimizer = importlib.import_module("common.optimizer").SGD(lr=0.01)
        (x_train, t_train), _ = load_mnist(normalize=True)
        x_train, t_train = x_train[:1000], t_train[:1000]
        np.random.seed(1)
        scale = float(SCALES[scale_index])
        # Preserve the source's initialization order (BN first, normal second).
        bn_network = network_cls(
            784, [100] * 5, 10, weight_init_std=scale, use_batchnorm=True
        )
        normal_network = network_cls(784, [100] * 5, 10, weight_init_std=scale)
        network = bn_network if batchnorm else normal_network
        rows = []
        measurements = OriginalMeasurements(output)
        with measurements.training():
            for epoch in range(20):
                for _ in range(1 if epoch == 0 else 10):
                    mask = np.random.choice(1000, 100)
                    x_batch, t_batch = x_train[mask], t_train[mask]
                    optimizer.update(
                        network.params, network.gradient(x_batch, t_batch)
                    )
                rows.append(
                    {
                        "update": epoch * 10 + 1,
                        "epoch": epoch,
                        "scale_index": scale_index + 1,
                        "weight_scale": scale,
                        "batchnorm": batchnorm,
                        "accuracy": float(network.accuracy(x_train, t_train)),
                    }
                )
        measurements.save(network.params, scope="source_training_and_evaluation")
    save_csv(output / "metrics.csv", rows)


TRIALS = tuple(
    Trial(
        f"dlfs1.ch06.batchnorm.scale-{index + 1:02d}."
        + ("bn-on" if batchnorm else "bn-off"),
        "numpy",
        {
            "scale_index": index + 1,
            "weight_scale": float(scale),
            "batchnorm": batchnorm,
            "epochs": 20,
        },
        COMMON_SOURCES + ("ch06/batch_norm_test.py",),
        lambda worktree, output, index=index, batchnorm=batchnorm: _run(
            index, batchnorm, worktree, output
        ),
    )
    for index, scale in enumerate(SCALES)
    for batchnorm in (False, True)
)
