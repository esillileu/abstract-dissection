"""Backfill DS1 full-train accuracy and train/test gap metrics in MLflow.

The command is a dry-run unless ``--apply`` is supplied. It never starts new
experiments; it evaluates checkpoints belonging to existing completed runs.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path

from deepscratch.core import configure_runtime
from deepscratch.datasets import load_mnist
from deepscratch.nn.layers import BatchNormalization
from deepscratch.trainer import ForwardTrainer

from dlfs.ds1.implemented.executor import (
    _model,
    _objective,
    _optimizer,
    _training_parameters,
)
from dlfs.ds1.implemented.final_gap import (
    TARGET_RUNS,
    TRAIN_FULL_ACCURACY,
    TRAIN_TEST_ACCURACY_GAP,
    evaluate_checkpoint_gap,
)
from dlfs.ds1.implemented.spec import parse_run_spec

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
TARGET_NAMES = {
    "e06": ("GT06", "CNN-SIMPLE-BOOK"),
    "e07": ("GT07", "CNN-DEEP-BOOK"),
    "e12": ("GT09", "MLP-EXT-ALL-BOOK"),
}


def latest_target_runs(client, *, targets: set[tuple[str, str]], seeds: set[int]):
    experiment = client.get_experiment_by_name("ds1")
    if experiment is None:
        raise ValueError("MLflow experiment does not exist: ds1")
    selected = {}
    for group_id, atomic_run_id in sorted(targets):
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=(
                "attributes.status = 'FINISHED' and "
                "tags.`run.type` = 'seed_trial' and "
                f"tags.`execution_group.id` = '{group_id}' and "
                f"tags.`atomic_run.id` = '{atomic_run_id}'"
            ),
            order_by=["attributes.start_time DESC"],
            max_results=5_000,
        )
        for run in runs:
            seed = int(
                run.data.params.get(
                    "seed/master",
                    run.data.params.get("seed", run.data.tags.get("master_seed", -1)),
                )
            )
            if seeds and seed not in seeds:
                continue
            selected.setdefault((group_id, atomic_run_id, seed), run)
    return list(selected.values())


def backfill_runs(
    client,
    runs,
    *,
    apply: bool,
    force: bool,
    evaluator: Callable[[object], dict[str, float]],
) -> dict[str, int]:
    counts = {"planned": 0, "updated": 0, "skipped": 0, "failed": 0}
    for run in runs:
        run_id = run.info.run_id
        atomic = run.data.tags.get("atomic_run.id", "")
        seed = run.data.params.get("seed/master", run.data.tags.get("master_seed", ""))
        present = {
            TRAIN_FULL_ACCURACY,
            TRAIN_TEST_ACCURACY_GAP,
        }.issubset(run.data.metrics)
        if present and not force:
            counts["skipped"] += 1
            print(f"skip {atomic} seed={seed} run={run_id}: metrics already exist")
            continue
        counts["planned"] += 1
        if not apply:
            print(f"would update {atomic} seed={seed} run={run_id}")
            continue
        try:
            metrics = evaluator(run)
            step = int(run.data.metrics.get("final/system/total_updates", 0))
            for key in (TRAIN_FULL_ACCURACY, TRAIN_TEST_ACCURACY_GAP):
                client.log_metric(run_id, key, float(metrics[key]), step=step)
            client.set_tag(run_id, "maintenance.full_train_gap", "checkpoint-v1")
        except Exception as exc:
            counts["failed"] += 1
            print(f"failed {atomic} seed={seed} run={run_id}: {exc}")
            continue
        counts["updated"] += 1
        print(
            f"updated {atomic} seed={seed} run={run_id}: "
            f"train_full={metrics[TRAIN_FULL_ACCURACY]:.6f} "
            f"gap={metrics[TRAIN_TEST_ACCURACY_GAP]:.6f}"
        )
    return counts


def evaluate_run(client, run, *, device: str) -> dict[str, float]:
    config = _run_config(run, device=device)
    backend, streams, _runtime = configure_runtime(config)
    dataset = _mapping(config, "dataset")
    if dataset.get("input_transform", "identity") not in {None, "identity"}:
        raise ValueError("backfill supports only identity input transforms")
    flatten = bool(dataset.get("flatten", True))
    (x_train, t_train), (x_test, t_test) = load_mnist(
        flatten=flatten,
        gpu=backend.is_gpu,
    )
    if (limit := dataset.get("train_limit")) is not None:
        x_train, t_train = x_train[: int(limit)], t_train[: int(limit)]
    if (limit := dataset.get("test_limit")) is not None:
        x_test, t_test = x_test[: int(limit)], t_test[: int(limit)]
    model = _model(
        _mapping(config, "model"),
        dropout_rng=backend.random_stream("dropout"),
    )
    if any(isinstance(layer, BatchNormalization) for layer in model.children()):
        model.forward(x_train[:1])
    objective = _objective(_mapping(config, "objective"), model.backend)
    optimizer = _optimizer(
        _mapping(config, "optimizer"),
        _training_parameters(model, objective),
    )
    loader = _mapping(config, "loader")
    trainer = ForwardTrainer(
        model,
        objective,
        optimizer,
        max_epochs=1,
        batch_size=int(loader.get("batch_size", 100)),
        drop_last=False,
        sampling_method="permutation_per_epoch",
        batch_rng=backend.random_stream("batch_order"),
    )
    checkpoint = resolve_run_checkpoint(client, run)
    _train, _test, metrics = evaluate_checkpoint_gap(
        trainer=trainer,
        model=model,
        checkpoint=checkpoint,
        x_train=x_train,
        t_train=t_train,
        x_test=x_test,
        t_test=t_test,
    )
    del streams
    return metrics


def resolve_run_checkpoint(client, run) -> Path:
    manifest_path = _download_file(
        client,
        run.info.run_id,
        "checkpoints/checkpoint_manifest.json",
    )
    candidates = []
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = manifest.get("latest") or manifest.get("final")
        if isinstance(checkpoint, dict) and checkpoint.get("path"):
            name = Path(str(checkpoint["path"])).name
            candidates.extend(
                (f"checkpoints/generations/{name}", f"checkpoints/{name}")
            )
    candidates.append("checkpoints/final.npz")
    for artifact_path in candidates:
        try:
            downloaded = Path(client.download_artifacts(run.info.run_id, artifact_path))
        except Exception:
            continue
        if downloaded.exists():
            return downloaded
    raise ValueError(f"latest checkpoint is unavailable for run {run.info.run_id}")


def _run_config(run, *, device: str) -> dict[str, object]:
    entrypoint = run.data.tags.get("code.entrypoint")
    if not entrypoint:
        raise ValueError(f"run {run.info.run_id} has no code.entrypoint tag")
    atomic = run.data.tags.get("atomic_run.id")
    config = parse_run_spec(entrypoint, atomic_run_id=atomic).to_executor_config()
    config["seed"] = int(
        run.data.params.get(
            "seed/master",
            run.data.params.get("seed", run.data.tags.get("master_seed", -1)),
        )
    )
    numerics = _mapping(config, "numerics")
    numerics["device"] = device
    numerics["backend"] = "cupy" if device.startswith("cuda") else "numpy"
    return config


def _download_file(client, run_id: str, artifact_path: str) -> Path | None:
    try:
        path = Path(client.download_artifacts(run_id, artifact_path))
    except Exception:
        return None
    return path if path.is_file() else None


def _mapping(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        action="append",
        choices=sorted(TARGET_NAMES),
        help="Repeat to select e06, e07, or e12; defaults to all.",
    )
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from mlflow.tracking import MlflowClient

    tracking_uri = (
        args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    targets = (
        {TARGET_NAMES[name] for name in args.target}
        if args.target
        else set(TARGET_RUNS)
    )
    runs = latest_target_runs(
        client,
        targets=targets,
        seeds=set(args.seed or ()),
    )
    counts = backfill_runs(
        client,
        runs,
        apply=args.apply,
        force=args.force,
        evaluator=lambda run: evaluate_run(client, run, device=args.device),
    )
    print(" ".join(f"{key}={value}" for key, value in counts.items()))
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
