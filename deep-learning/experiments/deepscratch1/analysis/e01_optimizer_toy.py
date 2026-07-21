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
DEFAULT_OUTPUT = Path("experiments/deepscratch1/results/analysis/e01_optimizer_toy_paths.png")


@dataclass(frozen=True)
class ToyPath:
    atomic_run_id: str
    mlflow_run_id: str
    steps: list[int]
    x: list[float]
    y: list[float]
    objective: list[float]


def objective_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(-10.0, 10.0, 0.01)
    y = np.arange(-5.0, 5.0, 0.01)
    grid_x, grid_y = np.meshgrid(x, y)
    z = grid_x * grid_x / 20.0 + grid_y * grid_y
    z[z > 7] = 0
    return grid_x, grid_y, z


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render e01 optimizer toy paths from MLflow runs.")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--mlflow-experiment", default=os.getenv("MLFLOW_EXPERIMENT_NAME", "deepscratch1"))
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

    # The book draws each point before its update. The run records the point
    # after each update, so prepend the declared initial point and omit the
    # final recorded point to recover the same 30 plotted positions.
    return ToyPath(
        atomic_run_id=atomic_run_id,
        mlflow_run_id=run_id,
        steps=list(range(len(steps))),
        x=[-7.0, *xs[:-1]],
        y=[2.0, *ys[:-1]],
        objective=[49.0 / 20.0 + 4.0, *objective[:-1]],
    )


def render_paths(paths: list[ToyPath], *, output: Path) -> None:
    grid_x, grid_y, z = objective_grid()
    paths_by_id = {path.atomic_run_id: path for path in paths}
    titles = {"TOY-SGD": "SGD", "TOY-MOM": "Momentum", "TOY-ADAGRAD": "AdaGrad", "TOY-ADAM": "Adam"}
    fig, axes = plt.subplots(nrows=2, ncols=2)
    for axis, atomic_run_id in zip(axes.flat, DEFAULT_ATOMIC_RUN_IDS, strict=True):
        path = paths_by_id[atomic_run_id]
        axis.plot(path.x, path.y, "o-", color="red", ms=2)
        final_x, final_y = path.x[-1], path.y[-1]
        axis.annotate(
            f"({final_x:.2f}, {final_y:.2f})",
            xy=(final_x, final_y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="red",
            fontsize=8,
        )
        axis.contour(grid_x, grid_y, z)
        axis.set_xlim(-10, 10)
        axis.set_ylim(-3, 3)
        axis.plot(0, 0, "+")
        axis.set_title(titles[atomic_run_id])
        axis.set_xlabel("x")
        axis.set_ylabel("y")

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
