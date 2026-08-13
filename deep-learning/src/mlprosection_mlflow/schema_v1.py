"""Schema-v1 MLflow projection for the portable experiment configuration."""

from __future__ import annotations

import csv
import json
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
    flatten_dict,
    make_condition_key,
    make_parent_group_key,
    make_run_key,
    parameter_manifest,
    pip_freeze,
    write_git_diff,
    write_json,
    write_metric_rows_csv,
    write_memory_history_csv,
    write_runtime_history_csv,
    write_text,
)


# Per-domain run material is deliberately kept outside MLflow's own artifact
# store. MLflow receives the lightweight record, while the local workspace is
# pleasant to browse and safe to clean independently.
ARTIFACT_ROOT = Path("exp")


def _storage_domain(value: object) -> str:
    """Make the tracking experiment safe to use as a local directory name."""
    name = str(value).strip()
    return (
        "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "-" for char in name
        )
        or "mlprosection"
    )


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


class SchemaV1Run:
    """Owns the full legacy-compatible record for one YAML experiment."""

    def __init__(self, config: dict[str, object]) -> None:
        self.config = normalize_config(config)
        self.seed = int(self.config["seed"])
        self.seeds = seed_config(self.seed)
        policy = _section(self.config, "policy")
        if policy.get("dataset_split_seed") is not None:
            self.seeds["dataset_split"] = int(policy["dataset_split_seed"])
        self.git_info = current_git_info(
            str(_section(self.config, "training")["entrypoint"])
        )
        self.condition = build_condition_config(self.config, self.git_info)
        self.identity = build_identity(self.config, self.condition, self.seeds)
        tracking = _section(self.config, "tracking")
        self.storage_domain = _storage_domain(
            tracking.get(
                "experiment", os.getenv("MLFLOW_EXPERIMENT_NAME", "mlprosection")
            )
        )
        domain_root = ARTIFACT_ROOT / self.storage_domain / "results"
        self.artifact_root = domain_root / "mlflow_artifacts" / self.identity.run_key
        self.local_checkpoint_root = domain_root / "checkpoints" / self.identity.run_key

    def runtime(self, *, model: Any | None = None) -> ExperimentRun:
        tracking = _section(self.config, "tracking")
        return ExperimentRun(
            options=RuntimeOptions(
                tracking_uri=(
                    os.getenv("MLFLOW_TRACKING_URI")
                    or str(
                        tracking.get(
                            "uri",
                            "http://127.0.0.1:5000",
                        )
                    )
                ),
                experiment_name=str(
                    tracking.get(
                        "experiment",
                        os.getenv("MLFLOW_EXPERIMENT_NAME", "mlprosection"),
                    )
                ),
                mlflow_enabled=bool(tracking.get("enabled", True)),
                upload_checkpoint=bool(tracking.get("upload_checkpoint", True)),
                upload_eval_checkpoints=bool(
                    tracking.get("upload_eval_checkpoints", True)
                ),
            ),
            run_name=f"{self.identity.atomic_run_id}-s{self.identity.master_seed:02d}",
            tags=build_tags(self.identity, self.config, self.git_info, model),
            params=flatten_dict({
                **self.condition,
                "seed": self.seeds,
                "policy": _section(self.config, "policy"),
                "regularization": _section(self.config, "regularization"),
            }),
        )

    def write_artifacts(
        self,
        *,
        model: Any | None,
        final_metrics: dict[str, float],
        metric_rows: list[tuple[int, str, float]],
        profiling_metrics: dict[str, int | float],
        reproducibility: dict[str, object] | None = None,
        evaluation_checkpoints: list[Path] | None = None,
    ) -> None:
        checkpoint_path, checkpoint_digest = _select_final_checkpoint(
            self.local_checkpoint_root,
            save_final=bool(
                _section(self.config, "checkpoint").get("save_final", True)
            ),
        )
        resolved = {
            **self.condition,
            "condition_key": self.identity.condition_key,
            "run_key": self.identity.run_key,
            "seed": self.seeds,
            "policy": _section(self.config, "policy"),
            "regularization": _section(self.config, "regularization"),
        }
        if reproducibility and "data" in reproducibility:
            resolved["runtime_data"] = reproducibility["data"]
        write_json(self.artifact_root / "config/resolved.json", resolved)
        write_json(self.artifact_root / "config/condition.json", self.condition)
        write_json(self.artifact_root / "config/seed.json", self.seeds)
        write_json(
            self.artifact_root / "config/profiling.json",
            _section(self.config, "profiling"),
        )
        write_json(
            self.artifact_root / "reproducibility/runtime.json", reproducibility or {}
        )
        write_json(self.artifact_root / "code/git.json", self.git_info)
        if self.git_info["dirty"]:
            write_git_diff(self.artifact_root / "code/git.diff.patch")
        write_text(self.artifact_root / "environment/python.txt", sys.version)
        write_text(self.artifact_root / "environment/packages.txt", pip_freeze())
        write_json(
            self.artifact_root / "environment/system.json", environment_artifacts()
        )
        backend = model.backend if model is not None else get_default_backend()
        write_json(
            self.artifact_root / "environment/backend.json",
            {
                "backend": backend.name,
                "device": backend.device,
                "dtype": backend.dtype_name,
            },
        )
        write_json(
            self.artifact_root / "environment/device.json", backend.memory_info()
        )
        write_json(
            self.artifact_root / "data/dataset_manifest.json",
            _section(self.config, "dataset"),
        )
        write_json(
            self.artifact_root / "model/architecture.json",
            _section(self.config, "model"),
        )
        write_text(
            self.artifact_root / "model/structure.txt",
            str(model) if model is not None else "",
        )
        write_json(
            self.artifact_root / "model/parameter_manifest.json",
            parameter_manifest(model) if model is not None else [],
        )
        write_json(
            self.artifact_root / "model/initialization_manifest.json",
            _section(self.config, "initializer"),
        )
        write_metric_rows_csv(
            self.artifact_root / "metrics/metrics.csv",
            run_key=self.identity.run_key,
            rows=metric_rows,
        )
        write_runtime_history_csv(
            self.artifact_root / "metrics/runtime_history.csv",
            build_runtime_history_rows(profiling_metrics),
        )
        write_memory_history_csv(
            self.artifact_root / "metrics/memory_history.csv",
            build_memory_history_rows(profiling_metrics),
        )
        write_json(self.artifact_root / "metrics/final.json", final_metrics)
        write_json(
            self.artifact_root / "profiles/profiling_summary.json",
            {
                "schema_version": 1,
                "enabled": _section(self.config, "profiling").get("enabled", False),
                "metrics": profiling_metrics,
            },
        )
        roles = {
            role: _checkpoint_role_manifest(self.local_checkpoint_root, role)
            for role in ("latest", "best")
        }
        _normalize_checkpoints_csv(self.artifact_root / "checkpoints.csv", roles)
        write_json(
            self.artifact_root / "checkpoints/checkpoint_manifest.json",
            {
                "format": "v2" if checkpoint_path else "none",
                "local_root": str(self.local_checkpoint_root.resolve()),
                # final remains a compatibility alias for latest.
                "final": None
                if checkpoint_path is None
                else {
                    "path": str(checkpoint_path.resolve()),
                    "epoch": _section(self.config, "training").get("max_epochs"),
                    "update": final_metrics.get("final/system/total_updates"),
                    "digest": checkpoint_digest,
                },
                "best": roles["best"],
                "latest": roles["latest"],
                "periodic": [],
                "epoch_checkpoints": [],
                "contains": {
                    "model": checkpoint_path is not None,
                    "optimizer": True,
                    "scheduler": False,
                    "rng_state": True,
                    "training_state": True,
                },
            },
        )


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
    backend = model.backend if model is not None else get_default_backend()
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
        "runtime.backend": backend.name,
        "runtime.device_type": "cuda" if backend.is_gpu else "cpu",
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
        "model.name": str(
            _section(config, "model").get(
                "name", ""
            )
        ),
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
    tags.update({
        str(key): str(value).format_map(substitutions)
        for key, value in declared_tags.items()
    })
    return tags


def _section(config: dict[str, object], name: str) -> dict[str, object]:
    value = config.get(name, {})
    assert isinstance(value, dict)
    return value


def _select_final_checkpoint(
    root: Path,
    *,
    save_final: bool,
) -> tuple[Path | None, str | None]:
    pointer = root / "latest.json"
    if not save_final or not pointer.exists():
        return None, None
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("only checkpoint schema version 2 is supported")
    path = root / str(payload["path"])
    return path, str(payload["sha256"])


def _checkpoint_role_manifest(root: Path, role: str) -> dict[str, object] | None:
    pointer = root / f"{role}.json"
    if not pointer.exists():
        return None
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    path = root / str(payload["path"])
    return {
        "path": str(path.resolve()),
        "epoch": int(payload["epoch"]),
        "update": int(payload["update"]),
        "digest": str(payload["sha256"]),
    }


def _normalize_checkpoints_csv(
    path: Path,
    roles: dict[str, dict[str, object] | None],
) -> None:
    """Keep the raw checkpoint index aligned with the retained semantic roles."""
    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    retained: list[dict[str, str]] = []
    for role, accepted_kinds in (
        ("latest", {"latest", "final"}),
        ("best", {"best", "selected"}),
    ):
        manifest = roles.get(role)
        if manifest is None:
            continue
        candidates = [row for row in rows if row.get("kind") in accepted_kinds]
        if candidates:
            row = dict(candidates[-1])
        else:
            row = {key: "" for key in fieldnames}
        row["kind"] = "selected" if role == "best" else "latest"
        row["path"] = str(manifest["path"])
        if "sha256" in fieldnames:
            row["sha256"] = str(manifest["digest"])
        retained.append(row)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(retained)


def _periodic_checkpoint_manifests(root: Path) -> list[dict[str, object]]:
    generations = root / "generations"
    if not generations.exists():
        return []
    output = []
    for path in sorted(generations.glob("periodic-*")):
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        output.append({
            "path": str(path.resolve()),
            "epoch": int(manifest["epoch"]),
            "update": int(manifest["global_step"]),
        })
    return output
