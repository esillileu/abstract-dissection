"""Backfill MLflow hierarchy, lifecycle tags, and profiler history for old runs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from mlprosection_mlflow.runtime import (
    build_profiling_history_rows,
    get_or_create_condition_parent,
    metric_batches,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--experiment", default="deepbase1")
    parser.add_argument("--apply", action="store_true", help="Write tags, parents, and missing profiler metrics.")
    args = parser.parse_args()

    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_uri)
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {args.experiment}")
    runs = client.search_runs([experiment.experiment_id], max_results=50_000)
    children = [run for run in runs if run.data.tags.get("run.type") == "seed_trial"]
    print(f"{args.experiment}: {len(children)} seed trials ({'apply' if args.apply else 'dry run'})")
    for child in children:
        tags = child.data.tags
        if not tags.get("condition.key"):
            print(f"skip {child.info.run_id}: missing condition.key")
            continue
        model_name = child.data.params.get("model/name")
        status = _trial_status(child.info.status)
        if not args.apply:
            print(f"would migrate {child.info.run_id} condition={tags['condition.key'][:12]} status={status}")
            continue
        parent_id = get_or_create_condition_parent(client, experiment_id=experiment.experiment_id, child_tags=tags)
        client.set_tag(child.info.run_id, "mlflow.parentRunId", parent_id)
        client.set_tag(child.info.run_id, "parent.mlflow_run_id", parent_id)
        client.set_tag(child.info.run_id, "trial.status", status)
        if model_name:
            client.set_tag(child.info.run_id, "model.name", model_name)
        _backfill_profiler_metrics(client, mlflow, child.info.run_id)


def _trial_status(status: str) -> str:
    return {"FINISHED": "finished", "FAILED": "failed", "KILLED": "killed"}.get(status, "running")


def _backfill_profiler_metrics(client, mlflow, run_id: str) -> None:
    run = client.get_run(run_id)
    if run.data.tags.get("migration.profiling_history.v1") == "complete":
        return
    try:
        path = client.download_artifacts(run_id, "profiles/profiling_summary.json")
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        rows = build_profiling_history_rows(metrics)
    except Exception as exc:
        client.set_tag(run_id, "migration.profiling_history.v1", f"skipped: {type(exc).__name__}")
        return
    for batch in metric_batches([(step, f"{step_type}/{metric}", value) for step_type, step, metric, value in rows], 1_000):
        client.log_batch(run_id, metrics=[mlflow.entities.Metric(key=key, value=value, timestamp=int(time.time() * 1000), step=step) for step, key, value in batch])
    client.set_tag(run_id, "migration.profiling_history.v1", "complete")


if __name__ == "__main__":
    main()
