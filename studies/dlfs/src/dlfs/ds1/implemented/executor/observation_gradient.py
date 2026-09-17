from __future__ import annotations

import sys
from dataclasses import asdict
from time import perf_counter

import numpy as np
from deepscratch.core import configure_runtime

from dlfs.ds1.implemented.adapters import (
    book_gradients,
    build_ds1_model,
    build_ds1_objective,
)
from dlfs.ds1.implemented.adapters import (
    load_ds1_mnist as _default_load_ds1_mnist,
)
from repro_core.context import ExperimentContext
from repro_core.context.contracts import ExperimentResult

from .common import _artifact_root, _mapping, _write_rows


def _get_load_ds1_mnist():
    mod = sys.modules.get("dlfs.ds1.implemented.executor")
    if mod is not None and hasattr(mod, "load_ds1_mnist"):
        return mod.load_ds1_mnist
    return _default_load_ds1_mnist


class GradientCheckObservationExecutor:
    """Reproduce ch05/gradient_check.py and retain timing and raw gradients."""

    def run(
        self, config: dict[str, object], context: ExperimentContext
    ) -> ExperimentResult:
        artifact_root = _artifact_root(context)
        observations = artifact_root / "observations"
        observations.mkdir(parents=True, exist_ok=True)
        backend, streams, actual_runtime = configure_runtime(config)
        context.metadata["runtime"] = actual_runtime
        context.metadata["seed_streams"] = asdict(streams)
        load_mnist = _get_load_ds1_mnist()
        (x_train, t_train), _ = load_mnist(
            flatten=True, one_hot_label=True, gpu=backend.is_gpu
        )
        sample_count = int(_mapping(config, "dataset").get("sample_count", 3))
        x_batch, t_batch = x_train[:sample_count], t_train[:sample_count]
        model = build_ds1_model(_mapping(config, "model"))
        objective = build_ds1_objective(_mapping(config, "objective"), model.backend)
        progress = context.metadata.get("progress_reporter")
        if progress is not None:
            progress.set_total_updates(2)

        backend.synchronize()
        started = perf_counter()
        model.numerical_gradient(x_batch, t_batch, objective)
        backend.synchronize()
        numerical_s = perf_counter() - started
        numerical = book_gradients(model)
        if progress is not None:
            progress.advance_to(1)

        model.zero_grad()
        backend.synchronize()
        started = perf_counter()
        objective.forward(model.forward(x_batch), t_batch)
        model.backward(objective.backward())
        backend.synchronize()
        backprop_s = perf_counter() - started
        backprop = book_gradients(model)
        if progress is not None:
            progress.advance_to(2)

        rows = []
        arrays = {}
        for name in ("W1", "b1", "W2", "b2"):
            numerical_array = numerical[name]
            backprop_array = backprop[name]
            difference = np.abs(backprop_array - numerical_array)
            rows.append(
                {
                    "parameter": name,
                    "mean_absolute_difference": float(difference.mean()),
                    "max_absolute_difference": float(difference.max()),
                    "numerical_mean_absolute_gradient": float(
                        np.abs(numerical_array).mean()
                    ),
                    "backprop_mean_absolute_gradient": float(
                        np.abs(backprop_array).mean()
                    ),
                }
            )
            arrays[f"numerical__{name}"] = numerical_array
            arrays[f"backprop__{name}"] = backprop_array
        timing_rows = [
            {"method": "numerical", "seconds": numerical_s},
            {"method": "backprop", "seconds": backprop_s},
            {"method": "speedup", "seconds": numerical_s / backprop_s},
        ]
        _write_rows(observations / "gradient_check.csv", rows, list(rows[0]))
        _write_rows(
            observations / "gradient_timing.csv", timing_rows, ["method", "seconds"]
        )
        np.savez_compressed(observations / "gradients.npz", **arrays)
        metrics = {
            "final/status/success": 1.0,
            "final/system/total_updates": 0.0,
            "final/system/completed_epochs": 0.0,
            "final/system/samples_seen": float(sample_count),
            "gradient_check/numerical_s": numerical_s,
            "gradient_check/backprop_s": backprop_s,
            "gradient_check/speedup": numerical_s / backprop_s,
            **{
                f"gradient_check/{row['parameter']}/mean_absolute_difference": float(
                    row["mean_absolute_difference"]
                )
                for row in rows
            },
        }
        metric_rows = [
            (
                0,
                f"observation/gradient_check/{row['parameter']}/mean_absolute_difference",
                float(row["mean_absolute_difference"]),
            )
            for row in rows
        ]
        metric_rows.extend(
            (0, f"observation/gradient_check/{key}", value)
            for key, value in (
                ("numerical_s", numerical_s),
                ("backprop_s", backprop_s),
                ("speedup", numerical_s / backprop_s),
            )
        )
        return ExperimentResult(
            metrics=metrics,
            artifact_root=artifact_root,
            model=model,
            metric_rows=tuple(metric_rows),
        )
