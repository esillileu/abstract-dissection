"""RunSpec pieces shared by experiment domains."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunIdentity:
    experiment_id: str
    group_id: str
    protocol: str
    recipe_id: str
    structure_signature: str


def load_variant(
    path: Path,
    *,
    atomic_run_id: str | None,
    overrides: dict[str, object] | None,
) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid YAML object: {path}")
    variants = value.pop("variants", None)
    if not isinstance(variants, dict) or not variants:
        raise ValueError(f"RunSpec YAML needs variants: {path}")
    if atomic_run_id is None:
        raise ValueError("this YAML defines variants; choose --atomic-run-id")
    selected = variants.get(atomic_run_id)
    if not isinstance(selected, dict):
        raise ValueError(f"unknown atomic_run_id: {atomic_run_id}")
    value = merge(value, selected)
    value["atomic_run_id"] = atomic_run_id
    if overrides:
        value = merge(value, overrides)
    return value


def mapping(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return dict(value)


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, object]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
