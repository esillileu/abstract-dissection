from __future__ import annotations

import numpy as np

from dlfs.ds1.implemented.adapters import activation_fn, initializer_scale
from repro_core.context import ExperimentContext
from repro_core.context.contracts import ExperimentResult

from ..records import DS1Records
from .common import _artifact_root, _mapping, _write_rows


class ActivationObservationExecutor:
    def run(
        self, config: dict[str, object], context: ExperimentContext
    ) -> ExperimentResult:
        artifact_root = _artifact_root(context)
        observations = artifact_root / "observations"
        observations.mkdir(parents=True, exist_ok=True)
        model_config = _mapping(config, "model")
        seed = int(_mapping(config, "dataset").get("input_seed", 40402))
        model_seed = int(model_config.get("model_seed", 40403))
        rng = np.random.default_rng(seed)
        weight_rng = np.random.default_rng(model_seed)
        x = rng.normal(size=(1000, 100))
        activation = str(model_config.get("activation", "relu"))
        initializer = str(model_config.get("initializer", "he"))
        hist_rows = []
        summary_rows = []
        for layer in range(1, int(model_config.get("depth", 5)) + 1):
            w = weight_rng.normal(
                scale=initializer_scale(initializer, x.shape[1]),
                size=(x.shape[1], int(model_config.get("width", 100))),
            )
            x = activation_fn(x @ w, activation)
            hist, edges = np.histogram(x, bins=50)
            for index, count in enumerate(hist):
                hist_rows.append(
                    {
                        "layer": layer,
                        "bin_index": index,
                        "bin_left": edges[index],
                        "bin_right": edges[index + 1],
                        "count": int(count),
                        "sample_count": int(x.size),
                    }
                )
            summary_rows.append(
                {
                    "layer": layer,
                    "mean": float(x.mean()),
                    "std": float(x.std()),
                    "min": float(x.min()),
                    "max": float(x.max()),
                    "zero_ratio": float((x == 0).mean()),
                    "sample_count": int(x.size),
                }
            )
        _write_rows(
            observations / "activation_histogram.csv",
            hist_rows,
            ["layer", "bin_index", "bin_left", "bin_right", "count", "sample_count"],
        )
        _write_rows(
            observations / "activation_summary.csv",
            summary_rows,
            ["layer", "mean", "std", "min", "max", "zero_ratio", "sample_count"],
        )
        records = DS1Records()
        records.bind_artifact_root(artifact_root)
        records.flush()
        return ExperimentResult(
            metrics={
                "final/status/success": 1.0,
                "final/system/total_updates": 0.0,
                "final/system/completed_epochs": 0.0,
                "final/system/samples_seen": 1000.0,
            },
            artifact_root=artifact_root,
            metric_rows=tuple(
                (
                    0,
                    f"observation/activation/layer_{row['layer']}/{key}",
                    float(row[key]),
                )
                for row in summary_rows
                for key in ("mean", "std", "zero_ratio")
            ),
        )
