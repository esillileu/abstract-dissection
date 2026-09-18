"""Artifact result manifest generation for schema-v1."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..runtime import write_json


def write_result_manifest(
    artifact_root: Path,
    checkpoint_paths: dict[str, Path] | None = None,
) -> Path:
    """Write the immutable file inventory uploaded with a SchemaV1 result."""
    manifest_path = artifact_root / "result_manifest.json"
    files = []
    for path in sorted(item for item in artifact_root.rglob("*") if item.is_file()):
        if path == manifest_path:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(
            {
                "path": path.relative_to(artifact_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    checkpoints = []
    for role, checkpoint in sorted((checkpoint_paths or {}).items()):
        sources = (
            [checkpoint]
            if checkpoint.is_file()
            else sorted(item for item in checkpoint.rglob("*") if item.is_file())
        )
        for source in sources:
            suffix = (
                source.name
                if checkpoint.is_file()
                else (
                    f"generations/{checkpoint.name}/"
                    f"{source.relative_to(checkpoint).as_posix()}"
                )
            )
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            checkpoints.append(
                {
                    "role": role,
                    "path": f"checkpoints/{suffix}",
                    "size": source.stat().st_size,
                    "sha256": digest,
                }
            )
    write_json(
        manifest_path,
        {
            "schema_name": "mlprosection-schema-v1",
            "schema_version": 1,
            "files": files,
            "checkpoints": checkpoints,
        },
    )
    return manifest_path
