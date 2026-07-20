"""Move legacy local MLflow staging trees into their experiment-domain directories."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import mlflow
from mlflow.entities import ViewType


LEGACY_ROOT = Path("experiments/results/mlflow_artifacts")
RESULTS_ROOT = Path("experiments/results")


def _domain(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in name) or "mlprosection"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="perform the moves; otherwise only print them")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--experiment", action="append", default=["deepbase1", "deepbase2"])
    args = parser.parse_args()
    client = mlflow.tracking.MlflowClient(args.tracking_uri)
    moved = 0
    for experiment_name in args.experiment:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"skip {experiment_name}: MLflow experiment not found")
            continue
        for run in client.search_runs([experiment.experiment_id], run_view_type=ViewType.ALL, max_results=50_000):
            run_key = run.data.tags.get("run.key")
            if not run_key:
                continue
            source = LEGACY_ROOT / run_key
            target = RESULTS_ROOT / _domain(experiment_name) / "mlflow_artifacts" / run_key
            if not source.is_dir() or target.exists():
                continue
            print(f"{source} -> {target}")
            if args.apply:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(target))
            moved += 1
    print(f"moved={moved}")


if __name__ == "__main__":
    main()
