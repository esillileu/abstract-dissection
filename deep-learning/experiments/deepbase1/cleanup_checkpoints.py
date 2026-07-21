"""Keep one local final checkpoint per completed deepbase1 run and remove checkpoint artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import mlflow
from mlflow.entities import ViewType


ROOT = Path("experiments")
DOMAIN_ROOT = ROOT / "deepbase1/results"
LOCAL_ROOT = DOMAIN_ROOT / "checkpoints"
REMOTE_ROOT = ROOT / "data/artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    args = parser.parse_args()
    client = mlflow.tracking.MlflowClient(args.tracking_uri)
    experiment = client.get_experiment_by_name("deepbase1")
    if experiment is None:
        raise ValueError("MLflow experiment not found: deepbase1")
    # Parent consolidation can leave old seed trials in MLflow's deleted
    # lifecycle stage. Their file artifacts still consume disk, so cleanup
    # must include them as well.
    runs = client.search_runs([experiment.experiment_id], run_view_type=ViewType.ALL, max_results=50_000)
    count = 0
    for run in runs:
        if run.data.tags.get("run.type") != "seed_trial" or run.info.status != "FINISHED":
            continue
        # Pre-schema runs did not have a deterministic run key. Their MLflow
        # run id is still stable and keeps the preserved final model isolated.
        run_key = run.data.tags.get("run.key") or f"legacy-{run.info.run_id}"
        source = DOMAIN_ROOT / "mlflow_artifacts" / run_key / "checkpoints" / "final.npz"
        target = LOCAL_ROOT / run_key / "final.npz"
        remote = REMOTE_ROOT / str(experiment.experiment_id) / run.info.run_id / "artifacts" / "checkpoints"
        print(f"{run.info.run_id} {source} -> {target}")
        if not args.apply:
            count += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            shutil.copy2(source, target)
        elif remote.is_dir():
            epochs = sorted(remote.glob("epoch-*/model.npz"))
            if epochs:
                shutil.copy2(epochs[-1], target)
        if not target.is_file():
            print(f"skip {run.info.run_id}: no final model found")
            continue
        manifest = {"local_root": str(target.parent.resolve()), "final": {"path": str(target.resolve())}, "epoch_checkpoints": []}
        # The local MLflow server is configured with this filesystem artifact
        # root. Keep the manifest, but remove only checkpoint payloads directly
        # instead of issuing one HTTP request per historical run.
        if remote.is_dir():
            for payload in remote.iterdir():
                if payload.name == "checkpoint_manifest.json":
                    continue
                if payload.is_dir():
                    shutil.rmtree(payload)
                else:
                    payload.unlink()
            (remote / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        shutil.rmtree(source.parent, ignore_errors=True)
        count += 1
    print(f"processed={count}")


if __name__ == "__main__":
    main()
