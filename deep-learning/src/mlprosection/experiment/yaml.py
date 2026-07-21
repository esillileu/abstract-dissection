from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from typing import Any


def load_yaml(
    path: str | Path,
    *,
    atomic_run_id: str | None = None,
    overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    import yaml

    with Path(path).open(encoding="utf-8") as file:
        value: Any = yaml.safe_load(file)
    if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
        raise ValueError("experiment YAML requires a string 'kind'")
    variants = value.pop("variants", None)
    if variants is not None:
        if not isinstance(variants, dict):
            raise ValueError("variants must be a mapping")
        if atomic_run_id is None:
            raise ValueError("this YAML defines variants; choose --atomic-run-id")
        selected = variants.get(atomic_run_id)
        if not isinstance(selected, dict):
            raise ValueError(f"unknown atomic_run_id: {atomic_run_id}")
        value = _merge(value, selected)
    elif atomic_run_id is not None and str(value.get("atomic_run_id")) != atomic_run_id:
        raise ValueError(f"YAML defines atomic_run_id {value.get('atomic_run_id')!r}, not {atomic_run_id!r}")
    if overrides:
        value = _merge(value, overrides)
    return value


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, object]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
