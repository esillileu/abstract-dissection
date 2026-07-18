from __future__ import annotations

import time

import numpy as np

from common import (
    RunSpec,
    basic_profiling_metrics,
    checkpoint_config,
    common_policy,
    log_trial,
    parser,
    profiling_config_dict,
    profiling_config_from_env,
)


ACTIVATIONS = {"SIG": "sigmoid", "TANH": "tanh", "RELU": "relu"}
INITIALIZERS = {
    "STD1": ("normal_std_1", 1.0),
    "STD001": ("normal_std_0.01", 0.01),
    "XAVIER": ("xavier", None),
    "HE": ("he", None),
}
ATOMIC_RUNS = {
    f"ACT-{prefix}-{suffix}": {"activation": activation, "initializer": name, "scale": scale}
    for prefix, activation in ACTIVATIONS.items()
    for suffix, (name, scale) in INITIALIZERS.items()
}


def activation_forward(name: str, x):
    if name == "sigmoid":
        return 1 / (1 + np.exp(-x))
    if name == "tanh":
        return np.tanh(x)
    if name == "relu":
        return np.maximum(0, x)
    raise ValueError(name)


def weight(width: int, initializer: str, scale: float | None):
    if initializer == "xavier":
        std = (1 / width) ** 0.5
    elif initializer == "he":
        std = (2 / width) ** 0.5
    else:
        std = float(scale)
    return np.random.randn(width, width).astype("f") * std


def stats(values, activation: str) -> dict[str, float]:
    saturation = 0.0
    if activation == "sigmoid":
        saturation = float(((values < 0.01) | (values > 0.99)).mean())
    elif activation == "tanh":
        saturation = float((np.abs(values) > 0.99).mean())
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p01": float(np.percentile(values, 1)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p99": float(np.percentile(values, 99)),
        "zero_ratio": float((values == 0).mean()),
        "saturation_ratio": saturation,
        "nonfinite_ratio": float((~np.isfinite(values)).mean()),
    }


def run_probe(config: dict[str, object], seed: int):
    np.random.seed(seed)
    x = np.random.randn(1000, 100).astype("f")
    first_std = None
    final_metrics = {
        "final/status/success": 1.0,
        "final/status/nan_detected": 0.0,
        "final/status/inf_detected": 0.0,
        "final/status/diverged": 0.0,
        "final/system/total_updates": 0.0,
        "final/system/completed_epochs": 0.0,
        "final/system/samples_seen": float(x.shape[0]),
    }
    rows = []
    max_saturation = 0.0
    max_zero = 0.0
    mean_shift = 0.0
    for index in range(5):
        w = weight(100, str(config["initializer"]), config["scale"])
        x = activation_forward(str(config["activation"]), x @ w)
        layer_stats = stats(x, str(config["activation"]))
        if first_std is None:
            first_std = layer_stats["std"]
        max_saturation = max(max_saturation, layer_stats["saturation_ratio"])
        max_zero = max(max_zero, layer_stats["zero_ratio"])
        mean_shift += abs(layer_stats["mean"])
        for name, value in layer_stats.items():
            metric = f"layer/{index + 1:02d}/activation/{name}"
            final_metrics[metric] = value
            rows.append(("step", index, metric, value))
    final_metrics["final/activation/std_retention_ratio"] = float(layer_stats["std"] / first_std) if first_std else 0.0
    final_metrics["final/activation/mean_absolute_shift"] = mean_shift / 5
    final_metrics["final/activation/max_saturation_ratio"] = max_saturation
    final_metrics["final/activation/max_zero_ratio"] = max_zero
    return rows, final_metrics


def build_spec(atomic_run_id: str, profiling: dict[str, object]) -> RunSpec:
    config = ATOMIC_RUNS[atomic_run_id]
    return RunSpec(
        atomic_run_id=atomic_run_id,
        experiment_ids=("e03",),
        execution_group_id="g03",
        recipe_id="RC-ACT-PROBE",
        structure_signature="activation-probe-100x5-v1",
        dataset={"id": "DS-SYNTH-ACT", "name": "standard_normal", "train_size": 1000, "input_shape": [100]},
        loader={"batch_size": 1000, "shuffle": False, "drop_last": False, "sampling_method": "synthetic_seeded", "steps_per_epoch": 1, "samples_per_epoch": 1000},
        model={"name": "ActivationProbe", "family": "probe", "task_type": "activation_probe", "input_shape": [100], "hidden_sizes": [100, 100, 100, 100, 100], "num_hidden_layers": 5, "activation": config["activation"], "structure_signature": "activation-probe-100x5-v1"},
        initializer={"name": config["initializer"], "scale": config["scale"], "seed": "seed/model_init"},
        optimizer={"name": "none"},
        scheduler={"name": "constant"},
        loss={"name": "none"},
        training={"max_epochs": 0, "max_updates": 0, "entrypoint": "experiments/run/e03_activation_probe.py"},
        evaluation={"primary_metric": "final/activation/std_retention_ratio"},
        numerics={"dtype": "float32", "backend": "numpy", "device": "cpu", "deterministic": True, "epsilon": 1e-7},
        checkpoint=checkpoint_config(),
        profiling=profiling,
        policy=common_policy(),
    )


def main() -> None:
    args = parser("Run e03 activation probe atomic trial.", sorted(ATOMIC_RUNS)).parse_args()
    profiling = profiling_config_dict(profiling_config_from_env())
    spec = build_spec(args.atomic_run_id, profiling)
    start = time.perf_counter()
    rows, final_metrics = run_probe(ATOMIC_RUNS[args.atomic_run_id], args.seed)
    end = time.perf_counter()
    final_metrics["runtime/train_total_s"] = end - start
    final_metrics["runtime/run_wall_total_s"] = end - start
    run_key = log_trial(
        spec=spec,
        seed=args.seed,
        tracking_uri=args.tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_enabled=not args.no_mlflow,
        final_metrics=final_metrics,
        history_rows=rows,
        profiling_metrics=basic_profiling_metrics(start, end),
    )
    print(f"run_key={run_key}")


if __name__ == "__main__":
    main()
