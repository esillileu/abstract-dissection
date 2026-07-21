from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ANALYSIS_ROOT = Path("experiments/deepbase1/results/analysis")
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_MLFLOW_EXPERIMENT = "deepbase1"


@dataclass(frozen=True)
class RunRecord:
    atomic_run_id: str
    mlflow_run_id: str
    run_name: str
    metrics: dict[str, float]
    params: dict[str, str]
    tags: dict[str, str]


@dataclass(frozen=True)
class Curve:
    steps: np.ndarray
    mean: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    run_count: int


def parser(description: str, default_output: Path) -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=description)
    argument_parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    argument_parser.add_argument(
        "--mlflow-experiment",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_MLFLOW_EXPERIMENT),
    )
    argument_parser.add_argument("--output", type=Path, default=default_output)
    argument_parser.add_argument("--summary-csv", type=Path, default=default_output.with_suffix(".csv"))
    argument_parser.add_argument("--run-id", action="append", default=[], help="MLflow run id to include. May be repeated.")
    return argument_parser


def import_mlflow():
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise RuntimeError("Install tracking dependencies with `uv sync --extra tracking`.") from exc
    return mlflow, MlflowClient


def client(tracking_uri: str):
    mlflow, client_cls = import_mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    return client_cls(tracking_uri=tracking_uri)


def latest_seeded_records(
    mlflow_client,
    *,
    experiment_name: str,
    atomic_run_ids: Iterable[str],
    param_filters: dict[str, str] | None = None,
) -> dict[str, list[RunRecord]]:
    experiment = mlflow_client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    wanted = list(atomic_run_ids)
    runs = mlflow_client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=5000,
    )
    selected = {}
    for run in runs:
        if run.info.status != "FINISHED" or run.data.tags.get("run.type") != "seed_trial":
            continue
        atomic_run_id = run.data.tags.get("atomic_run.id")
        if param_filters and any(run.data.params.get(key) != value for key, value in param_filters.items()):
            continue
        seed = run.data.params.get("seed/master", run.data.params.get("seed", run.info.run_id))
        key = (atomic_run_id, seed)
        if atomic_run_id in wanted and key not in selected:
            selected[key] = run

    grouped: dict[str, list[RunRecord]] = {atomic_run_id: [] for atomic_run_id in wanted}
    for (atomic_run_id, _), run in selected.items():
        grouped[atomic_run_id].append(_record_from_run(run))
    missing = [atomic_run_id for atomic_run_id in wanted if not grouped[atomic_run_id]]
    if missing:
        raise ValueError(f"missing MLflow runs for atomic run ids: {', '.join(missing)}")
    return grouped


def latest_run_ids_for_atomic_ids(mlflow_client, *, experiment_name: str, atomic_run_ids: Iterable[str], param_filters: dict[str, str] | None = None) -> list[str]:
    grouped = latest_seeded_records(
        mlflow_client,
        experiment_name=experiment_name,
        atomic_run_ids=atomic_run_ids,
        param_filters=param_filters,
    )
    return [record.mlflow_run_id for atomic_run_id in atomic_run_ids for record in grouped[atomic_run_id]]


def _record_from_run(run) -> RunRecord:
    return RunRecord(
        atomic_run_id=run.data.tags.get("atomic_run.id", run.info.run_name),
        mlflow_run_id=run.info.run_id,
        run_name=run.info.run_name,
        metrics={key: float(value) for key, value in run.data.metrics.items()},
        params={key: str(value) for key, value in run.data.params.items()},
        tags={key: str(value) for key, value in run.data.tags.items()},
    )


def load_records(mlflow_client, run_ids: Iterable[str]) -> list[RunRecord]:
    records = []
    for run_id in run_ids:
        run = mlflow_client.get_run(run_id)
        records.append(_record_from_run(run))
    return records


def metric_history(mlflow_client, *, run_id: str, key: str) -> tuple[list[int], list[float]]:
    history = sorted(mlflow_client.get_metric_history(run_id=run_id, key=key), key=lambda metric: metric.step)
    return [metric.step for metric in history], [float(metric.value) for metric in history]


def metric_curve(mlflow_client, records: Sequence[RunRecord], metric: str) -> Curve:
    histories = []
    for record in records:
        steps, values = metric_history(mlflow_client, run_id=record.mlflow_run_id, key=metric)
        if values:
            histories.append(dict(zip(steps, values, strict=True)))
    return curve_from_histories(histories)


def curve_from_histories(histories: Sequence[dict[int, float]]) -> Curve:
    if not histories:
        return Curve(np.array([]), np.array([]), np.array([]), np.array([]), 0)
    steps = sorted(set().union(*(history.keys() for history in histories)))
    values_by_step = [[history[step] for history in histories if step in history] for step in steps]
    return Curve(
        steps=np.asarray(steps),
        mean=np.asarray([np.mean(values) for values in values_by_step]),
        minimum=np.asarray([np.min(values) for values in values_by_step]),
        maximum=np.asarray([np.max(values) for values in values_by_step]),
        run_count=len(histories),
    )


def plot_curve(axis, value: Curve, *, label: str, linestyle: str = "-", marker: str | None = None) -> None:
    if not len(value.steps):
        return
    errors = np.vstack((value.mean - value.minimum, value.maximum - value.mean))
    axis.errorbar(value.steps, value.mean, yerr=errors, linestyle=linestyle, marker=marker, markersize=3, capsize=2, linewidth=1.6, label=f"{label} (n={value.run_count})")


def save_summary_csv(
    path: Path,
    *,
    records: list[RunRecord],
    metric_keys: list[str],
    param_keys: list[str] | None = None,
) -> None:
    param_keys = param_keys or []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["atomic_run_id", "mlflow_run_id", *param_keys, *metric_keys]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "atomic_run_id": record.atomic_run_id,
                "mlflow_run_id": record.mlflow_run_id,
            }
            row.update({key: record.params.get(key, "") for key in param_keys})
            row.update({key: record.metrics.get(key, "") for key in metric_keys})
            writer.writerow(row)


def print_outputs(output: Path, summary_csv: Path, records: list[RunRecord]) -> None:
    print(f"output={output}")
    print(f"summary_csv={summary_csv}")
    for record in records:
        print(f"{record.atomic_run_id} run_id={record.mlflow_run_id}")
