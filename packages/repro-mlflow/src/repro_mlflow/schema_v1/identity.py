"""Identity and condition configuration models and builders for schema-v1."""

from __future__ import annotations

import os
import sys
from typing import Any

from repro_core.config import normalize_config

from ..runtime import (
    RunIdentity,
    make_condition_key,
    make_parent_group_key,
    make_run_key,
)


def _storage_domain(value: object) -> str:
    """Make the tracking experiment safe to use as a local directory name."""
    name = str(value).strip()
    return (
        "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "-" for char in name
        )
        or "mlprosection"
    )


def _section(config: dict[str, object], name: str) -> dict[str, object]:
    value = config.get(name, {})
    assert isinstance(value, dict)
    return value


def seed_config(master_seed: int) -> dict[str, int]:
    return {
        "master": master_seed,
        "model_init": master_seed,
        "batch_order": master_seed + 10_000,
        "dropout": master_seed + 20_000,
        "negative_sampling": master_seed + 30_000,
        "synthetic_input": master_seed + 40_000,
        "dataset_split": master_seed,
        "worker": master_seed + 50_000,
    }


def build_condition_config(
    config: dict[str, object], git_info: dict[str, object]
) -> dict[str, object]:
    config = normalize_config(config)
    return {
        "schema_version": 1,
        "atomic_run_id": config["atomic_run_id"],
        "execution_group_id": config["execution_group_id"],
        "recipe_id": config["recipe_id"],
        "protocol_version": config.get("protocol_version", "legacy"),
        "structure_signature": config["structure_signature"],
        "code": {
            "git_commit": git_info["commit"],
            "git_diff_sha256": git_info["diff_sha256"],
            "entrypoint": git_info["entrypoint"],
        },
        **{
            key: _section(config, key)
            for key in (
                "dataset",
                "loader",
                "model",
                "initializer",
                "optimizer",
                "scheduler",
                "loss",
                "training",
                "evaluation",
                "numerics",
                "checkpoint",
                "profiling",
            )
        },
    }


def build_identity(
    config: dict[str, object], condition: dict[str, object], seeds: dict[str, int]
) -> RunIdentity:
    return RunIdentity(
        schema_version=1,
        project_name="mlprosection",
        experiment_ids=tuple(config["experiment_ids"]),
        atomic_run_id=str(config["atomic_run_id"]),
        execution_group_id=str(config["execution_group_id"]),
        recipe_id=str(config["recipe_id"]),
        structure_signature=str(config["structure_signature"]),
        condition_key=make_condition_key(condition),
        run_key=make_run_key(condition, seeds),
        master_seed=seeds["master"],
    )


def build_tags(
    identity: RunIdentity,
    config: dict[str, object],
    git_info: dict[str, object],
    model: Any | None,
) -> dict[str, str]:
    backend = model.backend if model is not None else None
    numerics = _section(config, "numerics")
    backend_name = (
        backend.name if backend is not None else str(numerics.get("backend", "numpy"))
    )
    is_gpu = (
        backend.is_gpu
        if backend is not None
        else str(numerics.get("device", "cpu")).startswith("cuda:")
    )
    group_identity = {
        "experiment/ids": identity.experiment_ids,
        "execution_group/id": identity.execution_group_id,
        "recipe/id": identity.recipe_id,
        "structure/signature": identity.structure_signature,
        "atomic_run/id": identity.atomic_run_id,
    }
    tags = {
        "schema.version": "1",
        "project.name": "mlprosection",
        "run.type": "seed_trial",
        "code.git_commit": str(git_info["commit"]),
        "code.git_branch": str(git_info["branch"]),
        "code.git_dirty": str(git_info["dirty"]).lower(),
        "code.repository": str(git_info["repository"]),
        "code.entrypoint": str(git_info["entrypoint"]),
        "code.runner_version": "1",
        "runtime.backend": backend_name,
        "runtime.device_type": "cuda" if is_gpu else "cpu",
        "runtime.platform": os.uname().sysname.lower(),
        "runtime.python_version": sys.version.split()[0],
        "atomic_run.id": identity.atomic_run_id,
        "experiment.ids": ",".join(identity.experiment_ids),
        "execution_group.id": identity.execution_group_id,
        "recipe.id": identity.recipe_id,
        "protocol.version": str(config.get("protocol_version", "legacy")),
        "structure.signature": identity.structure_signature,
        "condition.key": identity.condition_key,
        "condition.group.key": make_parent_group_key(group_identity),
        "run.key": identity.run_key,
        "master_seed": str(identity.master_seed),
        "dataset.id": str(_section(config, "dataset").get("id", "")),
        "model.name": str(_section(config, "model").get("name", "")),
        "model.family": str(_section(config, "model").get("family", "")),
        "task.type": str(_section(config, "model").get("task_type", "classification")),
        "trial.status": "running",
        "trial.attempt": os.getenv("MLFLOW_TRIAL_ATTEMPT", "1"),
        "retry.of": os.getenv("MLFLOW_RETRY_OF", ""),
        "parent.mlflow_run_id": os.getenv("MLFLOW_PARENT_RUN_ID", ""),
    }
    declared_tags = _section(config, "tracking").get("tags", {})
    if not isinstance(declared_tags, dict):
        raise ValueError("tracking.tags must be a mapping")
    substitutions = {
        "atomic_run_id": identity.atomic_run_id,
        "condition_id": identity.atomic_run_id,
        "experiment_id": identity.experiment_ids[0],
    }
    tags.update(
        {
            str(key): str(value).format_map(substitutions)
            for key, value in declared_tags.items()
        }
    )
    return tags
