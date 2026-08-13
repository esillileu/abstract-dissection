"""Content-addressed result cache used by original experiment runners."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import numpy as np


SCHEMA_VERSION = 1
COMPLETE_MARKER = "COMPLETE"
MANIFEST = "manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    materialized = list(rows)
    if fieldnames is None:
        if not materialized:
            raise ValueError(f"fieldnames are required for an empty CSV: {path}")
        fieldnames = tuple(materialized[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    host_arrays = {name: to_host(value) for name, value in arrays.items()}
    np.savez_compressed(path, **host_arrays)


def to_host(value: object) -> np.ndarray:
    """Convert NumPy/CuPy-compatible values to a host ndarray."""
    if isinstance(value, (list, tuple)):
        return np.asarray([to_host(item) for item in value])
    try:
        import cupy as cp

        if isinstance(value, cp.ndarray):
            cp.cuda.get_current_stream().synchronize()
            return cp.asnumpy(value)
    except (ImportError, RuntimeError):
        pass
    return np.asarray(value)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def cache_is_valid(
    directory: Path,
    expected: Mapping[str, object] | None = None,
) -> bool:
    marker = directory / COMPLETE_MARKER
    manifest_path = directory / MANIFEST
    if not marker.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
    ):
        return False
    if expected is not None:
        for key, value in expected.items():
            if manifest.get(key) != value:
                return False
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            return False
        relative = artifact.get("path")
        digest = artifact.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            return False
        path = directory / relative
        if not path.is_file() or sha256_file(path) != digest:
            return False
    return True


def publish_result(
    staging: Path,
    target: Path,
    *,
    identity: Mapping[str, object],
) -> None:
    artifacts = []
    for path in sorted(staging.rglob("*")):
        if not path.is_file() or path.name in {MANIFEST, COMPLETE_MARKER}:
            continue
        artifacts.append(
            {
                "path": path.relative_to(staging).as_posix(),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    if not artifacts:
        raise RuntimeError(f"runner produced no artifacts: {identity}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        **identity,
        "artifacts": artifacts,
        "status": "complete",
    }
    (staging / MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (staging / COMPLETE_MARKER).write_text("complete\n", encoding="utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if target.exists():
        backup = target.with_name(f".{target.name}.previous")
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(target, backup)
    try:
        os.replace(staging, target)
    except BaseException:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def staging_directory(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
