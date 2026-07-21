"""Seed-aware MLflow aggregation for deepbase2 experiment figures."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ANALYSIS_ROOT = Path("experiments/deepbase2/results/analysis")
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_MLFLOW_EXPERIMENT = "deepbase2"


@dataclass(frozen=True)
class Curve:
    steps: np.ndarray
    mean: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    run_count: int


def parser(description: str, output: Path) -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=description)
    value.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    value.add_argument("--mlflow-experiment", default=os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_MLFLOW_EXPERIMENT))
    value.add_argument("--output", type=Path, default=output)
    value.add_argument("--summary-csv", type=Path, default=output.with_suffix(".csv"))
    return value


def client(tracking_uri: str):
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise RuntimeError("Install tracking dependencies with `uv sync --extra tracking`.") from exc
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def latest_seeded_runs(mlflow_client, experiment_name: str, atomic_run_ids: list[str]):
    experiment = mlflow_client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")
    wanted = set(atomic_run_ids)
    selected = {}
    runs = mlflow_client.search_runs([experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=50_000)
    for run in runs:
        if run.info.status != "FINISHED" or run.data.tags.get("run.type") != "seed_trial":
            continue
        atomic = run.data.tags.get("atomic_run.id")
        seed = run.data.params.get("seed/master", run.data.params.get("seed", run.info.run_id))
        key = (atomic, seed)
        if atomic in wanted and key not in selected:
            selected[key] = run
    grouped = defaultdict(list)
    for (atomic, _), run in selected.items():
        grouped[atomic].append(run)
    missing = [atomic for atomic in atomic_run_ids if atomic not in grouped]
    if missing:
        raise ValueError(f"missing completed runs: {', '.join(missing)}")
    return grouped


def curve(mlflow_client, runs, metric: str) -> Curve:
    histories = []
    for run in runs:
        values = {item.step: float(item.value) for item in mlflow_client.get_metric_history(run.info.run_id, metric)}
        if values:
            histories.append(values)
    if not histories:
        return Curve(np.array([]), np.array([]), np.array([]), np.array([]), 0)
    steps = sorted(set().union(*(set(values) for values in histories)))
    values_by_step = [[values[step] for values in histories if step in values] for step in steps]
    return Curve(
        np.asarray(steps),
        np.asarray([np.mean(values) for values in values_by_step]),
        np.asarray([np.min(values) for values in values_by_step]),
        np.asarray([np.max(values) for values in values_by_step]),
        len(histories),
    )


def plot_band(axis, value: Curve, *, label: str, marker: str | None = None) -> None:
    if not len(value.steps):
        return
    errors = np.vstack((value.mean - value.minimum, value.maximum - value.mean))
    axis.errorbar(value.steps, value.mean, yerr=errors, label=f"{label} (n={value.run_count})", marker=marker, markersize=3, capsize=2, linewidth=1.6)


def write_summary(path: Path, grouped, curves: dict[str, Curve], metric: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["atomic_run_id", "seed_runs", "metric", "final_mean", "final_min", "final_max"])
        writer.writeheader()
        for atomic, runs in grouped.items():
            value = curves[atomic]
            writer.writerow({"atomic_run_id": atomic, "seed_runs": len(runs), "metric": metric, "final_mean": value.mean[-1] if len(value.mean) else "", "final_min": value.minimum[-1] if len(value.minimum) else "", "final_max": value.maximum[-1] if len(value.maximum) else ""})
