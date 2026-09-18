"""Artifact materialization and serialization for schema-v1 runs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from repro_core.numerics import BackendConfig, make_backend

from ..runtime import (
    build_memory_history_rows,
    build_runtime_history_rows,
    environment_artifacts,
    parameter_manifest,
    pip_freeze,
    write_git_diff,
    write_json,
    write_memory_history_csv,
    write_metric_rows_csv,
    write_runtime_history_csv,
    write_text,
)
from .checkpoints import (
    _checkpoint_role_manifest,
    _normalize_checkpoints_csv,
    _select_final_checkpoint,
)
from .identity import _section
from .manifest import write_result_manifest


def write_run_artifacts(
    *,
    config: dict[str, object],
    condition: dict[str, object],
    seeds: dict[str, int],
    identity_condition_key: str,
    identity_run_key: str,
    git_info: dict[str, object],
    artifact_root: Path,
    local_checkpoint_root: Path,
    model: Any | None,
    final_metrics: dict[str, float],
    metric_rows: list[tuple[int, str, float]],
    profiling_metrics: dict[str, int | float],
    reproducibility: dict[str, object] | None = None,
    evaluation_checkpoints: list[Path] | None = None,
) -> None:
    """Materialize the full schema-v1 artifact directory tree."""
    checkpoint_path, checkpoint_digest = _select_final_checkpoint(
        local_checkpoint_root,
        save_final=bool(_section(config, "checkpoint").get("save_final", True)),
    )
    resolved = {
        **condition,
        "condition_key": identity_condition_key,
        "run_key": identity_run_key,
        "seed": seeds,
        "policy": _section(config, "policy"),
        "regularization": _section(config, "regularization"),
    }
    if reproducibility and "data" in reproducibility:
        resolved["runtime_data"] = reproducibility["data"]
    write_json(artifact_root / "config/resolved.json", resolved)
    write_json(artifact_root / "config/condition.json", condition)
    write_json(artifact_root / "config/seed.json", seeds)
    write_json(
        artifact_root / "config/profiling.json",
        _section(config, "profiling"),
    )
    serializable_reproducibility = {
        key: value
        for key, value in (reproducibility or {}).items()
        if not callable(value)
        and key
        not in {
            "progress_reporter",
            "record_eval_checkpoint",
            "checkpoint_root",
            "artifact_root",
        }
    }
    write_json(
        artifact_root / "reproducibility/runtime.json",
        serializable_reproducibility,
    )
    write_json(artifact_root / "code/git.json", git_info)
    if git_info["dirty"]:
        write_git_diff(artifact_root / "code/git.diff.patch")
    write_text(artifact_root / "environment/python.txt", sys.version)
    write_text(artifact_root / "environment/packages.txt", pip_freeze())
    write_json(artifact_root / "environment/system.json", environment_artifacts())
    if model is not None:
        backend = model.backend
    else:
        numerics = _section(config, "numerics")
        backend = make_backend(
            BackendConfig(
                device=str(numerics.get("device", "cpu")),
                dtype=str(numerics.get("dtype", "float32")),
                seed=int(config.get("seed", 0)),
            )
        )
    write_json(
        artifact_root / "environment/backend.json",
        {
            "backend": backend.name,
            "device": backend.device,
            "dtype": backend.dtype_name,
        },
    )
    write_json(artifact_root / "environment/device.json", backend.memory_info())
    write_json(
        artifact_root / "data/dataset_manifest.json",
        _section(config, "dataset"),
    )
    write_json(
        artifact_root / "model/architecture.json",
        _section(config, "model"),
    )
    write_text(
        artifact_root / "model/structure.txt",
        str(model) if model is not None else "",
    )
    write_json(
        artifact_root / "model/parameter_manifest.json",
        parameter_manifest(model) if model is not None else [],
    )
    write_json(
        artifact_root / "model/initialization_manifest.json",
        _section(config, "initializer"),
    )
    write_metric_rows_csv(
        artifact_root / "metrics/metrics.csv",
        run_key=identity_run_key,
        rows=metric_rows,
    )
    write_runtime_history_csv(
        artifact_root / "metrics/runtime_history.csv",
        build_runtime_history_rows(profiling_metrics),
    )
    write_memory_history_csv(
        artifact_root / "metrics/memory_history.csv",
        build_memory_history_rows(profiling_metrics),
    )
    write_json(artifact_root / "metrics/final.json", final_metrics)
    write_json(
        artifact_root / "profiles/profiling_summary.json",
        {
            "schema_version": 1,
            "enabled": _section(config, "profiling").get("enabled", False),
            "metrics": profiling_metrics,
        },
    )
    roles = {
        role: _checkpoint_role_manifest(local_checkpoint_root, role)
        for role in ("latest", "best")
    }
    _normalize_checkpoints_csv(artifact_root / "checkpoints.csv", roles)
    write_json(
        artifact_root / "checkpoints/checkpoint_manifest.json",
        {
            "format": "v2" if checkpoint_path else "none",
            "local_root": str(local_checkpoint_root.resolve()),
            # final remains a compatibility alias for latest.
            "final": None
            if checkpoint_path is None
            else {
                "path": str(checkpoint_path.resolve()),
                "epoch": _section(config, "training").get("max_epochs"),
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
    write_result_manifest(artifact_root)
