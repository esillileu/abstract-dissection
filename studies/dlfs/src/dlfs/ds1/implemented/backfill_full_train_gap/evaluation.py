from __future__ import annotations

import json
from pathlib import Path

from deepscratch.core import configure_runtime
from deepscratch.datasets import load_mnist
from deepscratch.nn.layers import BatchNormalization
from deepscratch.trainer import ForwardTrainer

from dlfs.ds1.implemented.adapters import (
    build_ds1_model,
    build_ds1_objective,
    build_ds1_optimizer,
    training_parameters,
)
from dlfs.ds1.implemented.final_gap import evaluate_checkpoint_gap
from dlfs.ds1.implemented.spec import parse_run_spec


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
    model = build_ds1_model(
        _mapping(config, "model"),
        dropout_rng=backend.random_stream("dropout"),
    )
    if any(isinstance(layer, BatchNormalization) for layer in model.children()):
        model.forward(x_train[:1])
    objective = build_ds1_objective(_mapping(config, "objective"), model.backend)
    optimizer = build_ds1_optimizer(
        _mapping(config, "optimizer"),
        training_parameters(model, objective),
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
