"""Schema-v1 MLflow projection for the portable experiment configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from mlprosection.core.backend import get_default_backend
from mlprosection.experiment import normalize_config

from .runtime import (
    ExperimentRun,
    RunIdentity,
    RuntimeOptions,
    build_memory_history_rows,
    build_runtime_history_rows,
    current_git_info,
    environment_artifacts,
    file_digest,
    flatten_dict,
    make_condition_key,
    make_run_key,
    parameter_manifest,
    pip_freeze,
    write_git_diff,
    write_history_csv,
    write_json,
    write_memory_history_csv,
    write_runtime_history_csv,
    write_text,
)


ARTIFACT_ROOT = Path("experiments/results/mlflow_artifacts")


def seed_config(master_seed: int) -> dict[str, int]:
    return {
        "master": master_seed, "model_init": master_seed, "batch_order": master_seed + 10_000,
        "dropout": master_seed + 20_000, "negative_sampling": master_seed + 30_000,
        "synthetic_input": master_seed + 40_000, "dataset_split": master_seed,
        "worker": master_seed + 50_000,
    }


class SchemaV1Run:
    """Owns the full legacy-compatible record for one YAML experiment."""

    def __init__(self, config: dict[str, object]) -> None:
        self.config = normalize_config(config)
        self.seed = int(self.config["seed"])
        self.seeds = seed_config(self.seed)
        self.git_info = current_git_info(str(_section(self.config, "training")["entrypoint"]))
        self.condition = build_condition_config(self.config, self.git_info)
        self.identity = build_identity(self.config, self.condition, self.seeds)
        self.artifact_root = ARTIFACT_ROOT / self.identity.run_key

    def runtime(self, *, model: Any | None = None) -> ExperimentRun:
        tracking = _section(self.config, "tracking")
        return ExperimentRun(
            options=RuntimeOptions(
                tracking_uri=str(tracking.get("uri", os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))),
                experiment_name=str(tracking.get("experiment", os.getenv("MLFLOW_EXPERIMENT_NAME", "mlprosection"))),
                mlflow_enabled=bool(tracking.get("enabled", True)),
                upload_checkpoint=bool(tracking.get("upload_checkpoint", False)),
            ),
            run_name=f"{self.identity.atomic_run_id}-s{self.identity.master_seed:02d}",
            tags=build_tags(self.identity, self.config, self.git_info, model),
            params=flatten_dict({
                **self.condition, "seed": self.seeds,
                "policy": _section(self.config, "policy"),
                "regularization": _section(self.config, "regularization"),
            }),
        )

    def write_artifacts(
        self, *, model: Any | None, final_metrics: dict[str, float],
        history_rows: list[tuple[str, int, str, float]],
        profiling_metrics: dict[str, int | float], reproducibility: dict[str, object] | None = None,
    ) -> None:
        checkpoint_path, checkpoint_digest = _save_checkpoint(self.artifact_root, model)
        resolved = {
            **self.condition, "condition_key": self.identity.condition_key,
            "run_key": self.identity.run_key, "seed": self.seeds,
            "policy": _section(self.config, "policy"),
            "regularization": _section(self.config, "regularization"),
        }
        write_json(self.artifact_root / "config/resolved.json", resolved)
        write_json(self.artifact_root / "config/condition.json", self.condition)
        write_json(self.artifact_root / "config/seed.json", self.seeds)
        write_json(self.artifact_root / "config/profiling.json", _section(self.config, "profiling"))
        write_json(self.artifact_root / "reproducibility/runtime.json", reproducibility or {})
        write_json(self.artifact_root / "code/git.json", self.git_info)
        if self.git_info["dirty"]:
            write_git_diff(self.artifact_root / "code/git.diff.patch")
        write_text(self.artifact_root / "environment/python.txt", sys.version)
        write_text(self.artifact_root / "environment/packages.txt", pip_freeze())
        write_json(self.artifact_root / "environment/system.json", environment_artifacts())
        backend = model.backend if model is not None else get_default_backend()
        write_json(self.artifact_root / "environment/backend.json", {"backend": backend.name, "device": backend.device, "dtype": backend.dtype_name})
        write_json(self.artifact_root / "environment/device.json", backend.memory_info())
        write_json(self.artifact_root / "data/dataset_manifest.json", _section(self.config, "dataset"))
        write_json(self.artifact_root / "model/architecture.json", _section(self.config, "model"))
        write_text(self.artifact_root / "model/structure.txt", str(model) if model is not None else "")
        write_json(self.artifact_root / "model/parameter_manifest.json", parameter_manifest(model) if model is not None else [])
        write_json(self.artifact_root / "model/initialization_manifest.json", _section(self.config, "initializer"))
        write_history_csv(self.artifact_root / "metrics/history.csv", run_key=self.identity.run_key, rows=history_rows)
        write_history_csv(self.artifact_root / "metrics/update_history.csv", run_key=self.identity.run_key, rows=[row for row in history_rows if row[0] == "update"])
        write_history_csv(self.artifact_root / "metrics/epoch_history.csv", run_key=self.identity.run_key, rows=[row for row in history_rows if row[0] == "epoch"])
        write_history_csv(self.artifact_root / "metrics/eval_history.csv", run_key=self.identity.run_key, rows=[row for row in history_rows if row[0] == "eval"])
        write_runtime_history_csv(self.artifact_root / "metrics/runtime_history.csv", build_runtime_history_rows(profiling_metrics))
        write_memory_history_csv(self.artifact_root / "metrics/memory_history.csv", build_memory_history_rows(profiling_metrics))
        write_json(self.artifact_root / "metrics/final.json", final_metrics)
        write_json(self.artifact_root / "profiles/profiling_summary.json", {"schema_version": 1, "enabled": _section(self.config, "profiling").get("enabled", False), "metrics": profiling_metrics})
        write_json(self.artifact_root / "checkpoints/checkpoint_manifest.json", {
            "format": "npz" if checkpoint_path else "none",
            "final": None if checkpoint_path is None else {"path": str(checkpoint_path), "epoch": _section(self.config, "training").get("max_epochs"), "update": final_metrics.get("final/system/total_updates"), "digest": checkpoint_digest},
            "best": None,
            "epoch_checkpoints": [path.name for path in sorted((self.artifact_root / "checkpoints").glob("epoch-*")) if path.is_dir()],
            "contains": {"model": checkpoint_path is not None, "optimizer": True, "scheduler": False, "rng_state": True, "training_state": True},
        })


def build_condition_config(config: dict[str, object], git_info: dict[str, object]) -> dict[str, object]:
    config = normalize_config(config)
    return {
        "schema_version": 1, "atomic_run_id": config["atomic_run_id"],
        "execution_group_id": config["execution_group_id"], "recipe_id": config["recipe_id"],
        "structure_signature": config["structure_signature"],
        "code": {"git_commit": git_info["commit"], "git_diff_sha256": git_info["diff_sha256"], "entrypoint": git_info["entrypoint"]},
        **{key: _section(config, key) for key in (
            "dataset", "loader", "model", "initializer", "optimizer", "scheduler", "loss",
            "training", "evaluation", "numerics", "checkpoint", "profiling",
        )},
    }


def build_identity(config: dict[str, object], condition: dict[str, object], seeds: dict[str, int]) -> RunIdentity:
    return RunIdentity(
        schema_version=1, project_name="mlprosection", experiment_ids=tuple(config["experiment_ids"]),
        atomic_run_id=str(config["atomic_run_id"]), execution_group_id=str(config["execution_group_id"]),
        recipe_id=str(config["recipe_id"]), structure_signature=str(config["structure_signature"]),
        condition_key=make_condition_key(condition), run_key=make_run_key(condition, seeds), master_seed=seeds["master"],
    )


def build_tags(identity: RunIdentity, config: dict[str, object], git_info: dict[str, object], model: Any | None) -> dict[str, str]:
    backend = model.backend if model is not None else get_default_backend()
    return {
        "schema.version": "1", "project.name": "mlprosection", "run.type": "seed_trial",
        "code.git_commit": str(git_info["commit"]), "code.git_branch": str(git_info["branch"]),
        "code.git_dirty": str(git_info["dirty"]).lower(), "code.repository": str(git_info["repository"]),
        "code.entrypoint": str(git_info["entrypoint"]), "code.runner_version": "1",
        "runtime.backend": backend.name, "runtime.device_type": "cuda" if backend.is_gpu else "cpu",
        "runtime.platform": os.uname().sysname.lower(), "runtime.python_version": sys.version.split()[0],
        "atomic_run.id": identity.atomic_run_id, "execution_group.id": identity.execution_group_id,
        "recipe.id": identity.recipe_id, "structure.signature": identity.structure_signature,
        "condition.key": identity.condition_key, "run.key": identity.run_key, "master_seed": str(identity.master_seed),
        "dataset.id": str(_section(config, "dataset").get("id", "")),
        "model.name": str(_section(config, "model").get("name", _section(config, "model").get("alias", ""))),
        "model.family": str(_section(config, "model").get("family", "")),
        "task.type": str(_section(config, "model").get("task_type", "classification")),
        "trial.status": "running", "trial.attempt": os.getenv("MLFLOW_TRIAL_ATTEMPT", "1"),
        "retry.of": os.getenv("MLFLOW_RETRY_OF", ""), "parent.mlflow_run_id": os.getenv("MLFLOW_PARENT_RUN_ID", ""),
    }


def _section(config: dict[str, object], name: str) -> dict[str, object]:
    value = config.get(name, {})
    assert isinstance(value, dict)
    return value


def _save_checkpoint(root: Path, model: Any | None) -> tuple[Path | None, str | None]:
    if model is None or not hasattr(model, "save_params_npz"):
        return None, None
    path = root / "checkpoints" / "final.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_params_npz(path)
    return path, file_digest(path)
