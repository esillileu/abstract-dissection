from __future__ import annotations

import math
import time

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


ATOMIC_RUNS = {
    "TOY-SGD": {"optimizer": "SGD", "lr": 0.95},
    "TOY-MOM": {"optimizer": "Momentum", "lr": 0.1, "momentum": 0.9},
    "TOY-ADAGRAD": {"optimizer": "AdaGrad", "lr": 1.5, "eps": 1e-7},
    "TOY-ADAM": {"optimizer": "Adam", "lr": 0.3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-7},
}


def objective(x: float, y: float) -> float:
    return x * x / 20.0 + y * y


def gradient(x: float, y: float) -> tuple[float, float]:
    return x / 10.0, 2.0 * y


def run_path(config: dict[str, float | str]) -> tuple[list[tuple[str, int, str, float]], dict[str, float]]:
    x, y = -7.0, 2.0
    path_length = 0.0
    x_changes = 0
    y_changes = 0
    last_dx = None
    last_dy = None
    rows = []
    state = {"vx": 0.0, "vy": 0.0, "hx": 0.0, "hy": 0.0, "mx": 0.0, "my": 0.0, "t": 0}

    for step in range(30):
        gx, gy = gradient(x, y)
        dx, dy = update_delta(config, state, gx, gy)
        x += dx
        y += dy
        step_distance = math.sqrt(dx * dx + dy * dy)
        path_length += step_distance
        if last_dx is not None:
            x_changes += int((last_dx < 0) != (dx < 0))
            y_changes += int((last_dy < 0) != (dy < 0))
        last_dx, last_dy = dx, dy
        distance = math.sqrt(x * x + y * y)
        for metric, value in {
            "opt/x": x,
            "opt/y": y,
            "opt/objective": objective(x, y),
            "opt/distance_to_optimum": distance,
            "opt/step_distance": step_distance,
            "opt/cumulative_path_length": path_length,
            "opt/x_direction_changed": float(x_changes),
            "opt/y_direction_changed": float(y_changes),
        }.items():
            rows.append(("step", step, metric, float(value)))

    final = {
        "final/status/success": 1.0,
        "final/status/nan_detected": 0.0,
        "final/status/inf_detected": 0.0,
        "final/status/diverged": 0.0,
        "final/system/total_updates": 30.0,
        "final/system/completed_epochs": 0.0,
        "final/system/samples_seen": 0.0,
        "final/opt/objective": objective(x, y),
        "final/opt/distance_to_optimum": math.sqrt(x * x + y * y),
        "final/opt/path_length": path_length,
        "final/opt/x_direction_changes": float(x_changes),
        "final/opt/y_direction_changes": float(y_changes),
    }
    return rows, final


def update_delta(config: dict[str, float | str], state: dict[str, float], gx: float, gy: float) -> tuple[float, float]:
    name = config["optimizer"]
    lr = float(config["lr"])
    if name == "SGD":
        return -lr * gx, -lr * gy
    if name == "Momentum":
        momentum = float(config["momentum"])
        state["vx"] = momentum * state["vx"] - lr * gx
        state["vy"] = momentum * state["vy"] - lr * gy
        return state["vx"], state["vy"]
    if name == "AdaGrad":
        eps = float(config["eps"])
        state["hx"] += gx * gx
        state["hy"] += gy * gy
        return -lr * gx / (math.sqrt(state["hx"]) + eps), -lr * gy / (math.sqrt(state["hy"]) + eps)
    if name == "Adam":
        beta1 = float(config["beta1"])
        beta2 = float(config["beta2"])
        eps = float(config["eps"])
        state["t"] += 1
        state["mx"] += (1 - beta1) * (gx - state["mx"])
        state["my"] += (1 - beta1) * (gy - state["my"])
        state["hx"] += (1 - beta2) * (gx * gx - state["hx"])
        state["hy"] += (1 - beta2) * (gy * gy - state["hy"])
        lr_t = lr * math.sqrt(1 - beta2 ** state["t"]) / (1 - beta1 ** state["t"])
        return -lr_t * state["mx"] / (math.sqrt(state["hx"]) + eps), -lr_t * state["my"] / (math.sqrt(state["hy"]) + eps)
    raise ValueError(f"unknown optimizer: {name}")


def build_spec(atomic_run_id: str, profiling: dict[str, object]) -> RunSpec:
    config = ATOMIC_RUNS[atomic_run_id]
    return RunSpec(
        atomic_run_id=atomic_run_id,
        experiment_ids=("e01",),
        execution_group_id="g01",
        recipe_id="RC-TOY-OPT",
        structure_signature="analytic-toy-v1",
        dataset={"id": "DS-TOY-ANALYTIC", "name": "analytic objective", "train_size": 0, "test_size": 0},
        loader={"batch_size": 1, "shuffle": False, "drop_last": False, "sampling_method": "deterministic", "steps_per_epoch": 30, "samples_per_epoch": 0},
        model={"name": "quadratic_objective", "family": "analytic", "task_type": "optimization", "input_shape": [2], "output_shape": [1], "structure_signature": "analytic-toy-v1"},
        initializer={"name": "constant", "initial_point": [-7.0, 2.0]},
        optimizer=config,
        scheduler={"name": "constant"},
        loss={"name": "quadratic", "reduction": "none"},
        training={"max_epochs": 0, "max_updates": 30, "entrypoint": "experiments/run/e01_optimizer_toy.py"},
        evaluation={"primary_metric": "final/opt/distance_to_optimum"},
        numerics={"dtype": "float64", "backend": "numpy", "device": "cpu", "deterministic": True, "epsilon": 1e-7},
        checkpoint=checkpoint_config(),
        profiling=profiling,
        policy=common_policy(seed_count=1),
    )


def main() -> None:
    args = parser("Run e01 optimizer toy atomic trial.", sorted(ATOMIC_RUNS)).parse_args()
    profiling = profiling_config_dict(profiling_config_from_env())
    spec = build_spec(args.atomic_run_id, profiling)
    start = time.perf_counter()
    history_rows, final_metrics = run_path(ATOMIC_RUNS[args.atomic_run_id])
    end = time.perf_counter()
    profiling_metrics = basic_profiling_metrics(start, end)
    final_metrics["runtime/train_total_s"] = end - start
    final_metrics["runtime/run_wall_total_s"] = end - start
    run_key = log_trial(
        spec=spec,
        seed=args.seed,
        tracking_uri=args.tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_enabled=not args.no_mlflow,
        final_metrics=final_metrics,
        history_rows=history_rows,
        profiling_metrics=profiling_metrics,
    )
    print(f"run_key={run_key}")


if __name__ == "__main__":
    main()
