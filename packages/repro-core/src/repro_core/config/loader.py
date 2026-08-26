"""YAML configuration loading and deep merging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    content = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"Invalid YAML config at {file_path}: expected mapping")
    return content


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
