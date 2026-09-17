from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from deepscratch.profiling.backend import create_device_timer

from repro_core.context import ExperimentContext


def _device_timer(config: dict[str, object], backend):
    profiling = _mapping(config, "profiling")
    return create_device_timer(
        backend, enabled=bool(profiling.get("device_timing", False))
    )


def _mapping(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _artifact_root(context: ExperimentContext) -> Path:
    root = context.metadata.get("artifact_root")
    if root is None:
        raise ValueError("experiment context is missing artifact_root")
    return Path(str(root))


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_digest(path: Path) -> str:
    if not path.exists():
        return ""
    if path.is_file():
        return _file_digest(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(_file_digest(child).encode())
    return digest.hexdigest()


def _write_rows(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
