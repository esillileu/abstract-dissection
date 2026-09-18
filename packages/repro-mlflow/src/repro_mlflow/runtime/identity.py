"""Stable identity structures and canonical hashing helpers for MLflow runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunIdentity:
    schema_version: int
    project_name: str
    experiment_ids: tuple[str, ...]
    atomic_run_id: str
    execution_group_id: str
    recipe_id: str
    structure_signature: str
    condition_key: str
    run_key: str
    master_seed: int


@dataclass(frozen=True)
class RuntimeOptions:
    tracking_uri: str
    experiment_name: str
    upload_checkpoint: bool = True
    upload_eval_checkpoints: bool = True
    queue_size: int = 256
    metric_batch_size: int = 1000


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def make_condition_key(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()


def make_run_key(condition: dict[str, Any], seed: dict[str, Any]) -> str:
    return hashlib.sha256(
        (canonical_json(condition) + canonical_json(seed)).encode()
    ).hexdigest()


def make_parent_group_key(params: dict[str, object]) -> str:
    """Hash the stable identity fields that define one seed-trial group."""
    stable = {
        key: value
        for key, value in params.items()
        if not key.startswith("code/") and not key.startswith("seed/")
    }
    return hashlib.sha256(canonical_json(stable).encode()).hexdigest()


def flatten_dict(value: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, child in value.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(child, dict):
            output.update(flatten_dict(child, prefix=name))
        else:
            output[name] = child
    return output


__all__ = [
    "RunIdentity",
    "RuntimeOptions",
    "canonical_json",
    "flatten_dict",
    "make_condition_key",
    "make_parent_group_key",
    "make_run_key",
]
