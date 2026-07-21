from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import latest_seeded_records


EXPERIMENT_ID = "e01"
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


@dataclass(frozen=True)
class AggregateToyPath:
    atomic_run_id: str
    steps: np.ndarray
    x_mean: np.ndarray
    x_minimum: np.ndarray
    x_maximum: np.ndarray
    y_mean: np.ndarray
    y_minimum: np.ndarray
    y_maximum: np.ndarray
    run_count: int


def objective_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-8.0, 8.0, 240)
    y = np.linspace(-3.0, 3.0, 180)
    grid_x, grid_y = np.meshgrid(x, y)
    z = grid_x * grid_x / 20.0 + grid_y * grid_y
    return grid_x, grid_y, z


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render e01 optimizer toy paths from MLflow runs.")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--mlflow-experiment", default=os.getenv("MLFLOW_EXPERIMENT_NAME", "deepbase1"))
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


def aggregate_paths(paths: list[ToyPath]) -> list[AggregateToyPath]:
    result = []
    for atomic_run_id in DEFAULT_ATOMIC_RUN_IDS:
        group = [path for path in paths if path.atomic_run_id == atomic_run_id]
        steps = sorted(set().union(*(path.steps for path in group)))
        def values(field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            series = [{step: value for step, value in zip(path.steps, getattr(path, field), strict=True)} for path in group]
            per_step = [[item[step] for item in series if step in item] for step in steps]
            return tuple(np.asarray([fn(items) for items in per_step]) for fn in (np.mean, np.min, np.max))
        x_mean, x_minimum, x_maximum = values("x")
        y_mean, y_minimum, y_maximum = values("y")
        result.append(AggregateToyPath(atomic_run_id, np.asarray(steps), x_mean, x_minimum, x_maximum, y_mean, y_minimum, y_maximum, len(group)))
    return result


def render_paths(paths: list[ToyPath], *, output: Path) -> None:
    grid_x, grid_y, z = objective_grid()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 9))
    for axis, path in zip(axes.flat, aggregate_paths(paths), strict=True):
        contour = axis.contour(grid_x, grid_y, z, levels=25, cmap="Greys", linewidths=0.7)
        axis.clabel(contour, inline=True, fontsize=7, fmt="%.1f")
        x_error = np.vstack((path.x_mean - path.x_minimum, path.x_maximum - path.x_mean))
        y_error = np.vstack((path.y_mean - path.y_minimum, path.y_maximum - path.y_mean))
        axis.errorbar(path.x_mean, path.y_mean, xerr=x_error, yerr=y_error, fmt="o-", markersize=3, capsize=2, linewidth=1.6, color="tab:red", label=f"mean (n={path.run_count})")
        axis.scatter([0.0], [0.0], marker="+", color="black", s=50, label="optimum")
        axis.set_title(path.atomic_run_id)
        axis.set_xlim(-10, 10)
        axis.set_ylim(-10, 10)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.legend(fontsize=7)

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
        grouped = latest_seeded_records(client, experiment_name=args.mlflow_experiment, atomic_run_ids=atomic_run_ids)
        run_ids = [record.mlflow_run_id for atomic_run_id in atomic_run_ids for record in grouped[atomic_run_id]]

    paths = [path_from_run(client=client, run_id=run_id) for run_id in run_ids]
    render_paths(paths=paths, output=args.output)
    print(f"output={args.output}")
    for path in paths:
        print(f"{path.atomic_run_id} run_id={path.mlflow_run_id}")


if __name__ == "__main__":
    main()
