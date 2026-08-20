"""Original ch05 gradient-check reproduction."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from exp.deepscratch.original_runtime.measurement import OriginalMeasurements
from exp.deepscratch.original_runtime.runtime_context import master_seed

from .common import (
    COMMON_SOURCES,
    Trial,
    importlib,
    np,
    save_csv,
    save_npz,
    save_params,
    source_imports,
)


SOURCE = "ch05/gradient_check.py"


def _run(worktree: Path, output: Path) -> None:
    with source_imports(worktree):
        load_mnist = importlib.import_module("dataset.mnist").load_mnist
        network_cls = importlib.import_module("ch05.two_layer_net").TwoLayerNet
        np.random.seed(master_seed())
        (x_train, t_train), _ = load_mnist(
            normalize=True, one_hot_label=True
        )
        network = network_cls(input_size=784, hidden_size=50, output_size=10)
        x_batch, t_batch = x_train[:3], t_train[:3]

        started = perf_counter()
        numerical = network.numerical_gradient(x_batch, t_batch)
        numerical_s = perf_counter() - started
        started = perf_counter()
        backprop = network.gradient(x_batch, t_batch)
        backprop_s = perf_counter() - started

        rows = []
        metric_rows = []
        arrays = {}
        for name in ("W1", "b1", "W2", "b2"):
            difference = np.abs(backprop[name] - numerical[name])
            row = {
                "parameter": name,
                "mean_absolute_difference": float(difference.mean()),
                "max_absolute_difference": float(difference.max()),
                "numerical_mean_absolute_gradient": float(np.abs(numerical[name]).mean()),
                "backprop_mean_absolute_gradient": float(np.abs(backprop[name]).mean()),
            }
            rows.append(row)
            metric_rows.extend((
                {"metric": f"gradient_check/{name}/mean_absolute_difference", "value": row["mean_absolute_difference"]},
                {"metric": f"gradient_check/{name}/max_absolute_difference", "value": row["max_absolute_difference"]},
            ))
            arrays[f"numerical__{name}"] = numerical[name]
            arrays[f"backprop__{name}"] = backprop[name]
        metric_rows.extend((
            {"metric": "gradient_check/numerical_s", "value": numerical_s},
            {"metric": "gradient_check/backprop_s", "value": backprop_s},
            {"metric": "gradient_check/speedup", "value": numerical_s / backprop_s},
        ))
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "gradient_check.csv", rows)
    save_csv(output / "metrics.csv", metric_rows)
    save_npz(output / "gradients.npz", **arrays)
    save_params(output / "checkpoint.npz", network.params)
    measurements = OriginalMeasurements(output)
    measurements.training_wall_time_s = numerical_s + backprop_s
    measurements.save(network.params, scope="gradient_check")
    (output / "gradient_timing.json").write_text(
        json.dumps(
            {
                "numerical_s": numerical_s,
                "backprop_s": backprop_s,
                "speedup": numerical_s / backprop_s,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


TRIALS = (
    Trial(
        "dlfs1.ch05.gradient-check",
        "numpy",
        {"samples": 3, "numerical_step": 1e-4, "updates": 0},
        COMMON_SOURCES + (SOURCE, "ch05/two_layer_net.py"),
        _run,
    ),
)
