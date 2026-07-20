from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ATOMIC_RUN_IDS = ("TOY-SGD", "TOY-MOM", "TOY-ADAGRAD", "TOY-ADAM")
DEFAULT_OUTPUT = Path("experiments/deepbase1/results/analysis/e01_optimizer_toy_paths.png")


@dataclass(frozen=True)
class ToyPath:
    atomic_run_id: str
    mlflow_run_id: str
    steps: list[int]
    x: list[float]
    y: list[float]
    objective: list[float]


def objective_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-8.0, 8.0, 240)
    y = np.linspace(-3.0, 3.0, 180)
    grid_x, grid_y = np.meshgrid(x, y)
    z = grid_x * grid_x / 20.0 + grid_y * grid_y
    return grid_x, grid_y, z


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render e01 optimizer toy paths from MLflow runs.")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--mlflow-experiment", default=os.getenv("MLFLOW_EXPERIMENT_NAME", "mlprosection"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", action="append", default=[], help="MLflow run id to include. May be repeated.")
    parser.add_argument(
        "--atomic-run-id",
        action="append",
        choices=DEFAULT_ATOMIC_RUN_IDS,
        default=[],
        help="Atomic run id to query when --run-id is not provided. May be repeated.",
    )
    return parser.parse_args()


def import_mlflow():
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise RuntimeError("Install tracking dependencies with `uv sync --extra tracking`.") from exc
    return mlflow, MlflowClient


def metric_values(client, *, run_id: str, metric_key: str) -> tuple[list[int], list[float]]:
    history = client.get_metric_history(run_id=run_id, key=metric_key)
    history = sorted(history, key=lambda metric: metric.step)
    return [metric.step for metric in history], [float(metric.value) for metric in history]


def path_from_run(client, *, run_id: str) -> ToyPath:
    run = client.get_run(run_id)
    atomic_run_id = run.data.tags.get("atomic_run.id", run.info.run_name)
    steps, xs = metric_values(client=client, run_id=run_id, metric_key="step/opt/x")
    y_steps, ys = metric_values(client=client, run_id=run_id, metric_key="step/opt/y")
    obj_steps, objective = metric_values(client=client, run_id=run_id, metric_key="step/opt/objective")

    if steps != y_steps or steps != obj_steps:
        raise ValueError(f"metric steps do not align for run {run_id}")
    if not steps:
        raise ValueError(f"run {run_id} has no e01 path metrics")

    return ToyPath(
        atomic_run_id=atomic_run_id,
        mlflow_run_id=run_id,
        steps=steps,
        x=xs,
        y=ys,
        objective=objective,
    )


def latest_run_ids_for_atomic_ids(
    client,
    *,
    experiment_name: str,
    atomic_run_ids: list[str],
) -> list[str]:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )
    selected: dict[str, str] = {}
    wanted = set(atomic_run_ids)
    for run in runs:
        atomic_run_id = run.data.tags.get("atomic_run.id")
        if atomic_run_id in wanted and atomic_run_id not in selected:
            selected[atomic_run_id] = run.info.run_id

    missing = [atomic_run_id for atomic_run_id in atomic_run_ids if atomic_run_id not in selected]
    if missing:
        raise ValueError(f"missing MLflow runs for atomic run ids: {', '.join(missing)}")
    return [selected[atomic_run_id] for atomic_run_id in atomic_run_ids]


def render_paths(paths: list[ToyPath], *, output: Path) -> None:
    grid_x, grid_y, z = objective_grid()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    contour = axes[0].contour(grid_x, grid_y, z, levels=25, cmap="Greys", linewidths=0.7)
    axes[0].clabel(contour, inline=True, fontsize=7, fmt="%.1f")
    for path in paths:
        axes[0].plot(path.x, path.y, marker="o", markersize=3, linewidth=1.8, label=path.atomic_run_id)
    axes[0].scatter([0.0], [0.0], marker="x", color="black", s=50, label="optimum")
    axes[0].set_title(r"Optimizer path on $f(x, y) = x^2 / 20 + y^2$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    for path in paths:
        axes[1].plot(path.steps, path.objective, marker="o", markersize=3, linewidth=1.8, label=path.atomic_run_id)
    axes[1].set_title("Objective by update")
    axes[1].set_xlabel("update")
    axes[1].set_ylabel("objective")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    mlflow, client_cls = import_mlflow()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = client_cls(tracking_uri=args.tracking_uri)

    run_ids = args.run_id
    if not run_ids:
        atomic_run_ids = args.atomic_run_id or list(DEFAULT_ATOMIC_RUN_IDS)
        run_ids = latest_run_ids_for_atomic_ids(
            client=client,
            experiment_name=args.mlflow_experiment,
            atomic_run_ids=atomic_run_ids,
        )

    paths = [path_from_run(client=client, run_id=run_id) for run_id in run_ids]
    render_paths(paths=paths, output=args.output)
    print(f"output={args.output}")
    for path in paths:
        print(f"{path.atomic_run_id} run_id={path.mlflow_run_id}")


if __name__ == "__main__":
    main()
