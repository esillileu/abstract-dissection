"""Artifact verification helpers for MLflow runs."""

from __future__ import annotations

import json
from pathlib import Path

from .environment import file_digest


def _verify_uploaded_manifest(client, run_id: str) -> None:
    """Verify every uploaded record file before declaring a run durable."""
    from ..artifact_cache import MlflowArtifactCache

    tracking_uri = getattr(client, "tracking_uri", None)
    if not tracking_uri:
        raise ValueError("manifest verification requires an explicit tracking URI")
    cache = MlflowArtifactCache(client, str(tracking_uri))
    staged: list[tuple[str, Path]] = []
    try:
        manifest_path = cache.fetch(run_id, "result_manifest.json")
        staged.append(("result_manifest.json", manifest_path))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or not isinstance(
            manifest.get("files"), list
        ):
            raise ValueError("invalid result manifest")
        for item in [*manifest["files"], *manifest.get("checkpoints", [])]:
            relative = str(item["path"])
            downloaded = cache.fetch(run_id, relative)
            staged.append((relative, downloaded))
            if file_digest(downloaded) != item["sha256"]:
                raise ValueError(f"artifact digest mismatch: {relative}")
        for relative, source in staged:
            cache.replace(run_id, relative, source)
    finally:
        for _, source in staged:
            cache.discard(source)


__all__ = ["_verify_uploaded_manifest"]
