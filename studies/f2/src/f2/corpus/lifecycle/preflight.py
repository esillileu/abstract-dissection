"""Preflight checks, environment inspection, and hash helpers."""

from __future__ import annotations

import hashlib
import json
import locale
import shutil
import subprocess
from pathlib import Path
from typing import Any

from repro_io.s3 import S3ObjectStore


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def config_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tool_versions() -> dict[str, str]:
    versions: dict[str, str] = {"locale": locale.setlocale(locale.LC_ALL, None)}
    for tool, args in {
        "zstd": ["--version"],
        "awk": ["--version"],
        "sed": ["--version"],
        "perl": ["-v"],
    }.items():
        try:
            line = subprocess.check_output(
                [tool, *args], text=True, stderr=subprocess.STDOUT
            ).splitlines()[0]
        except (OSError, subprocess.CalledProcessError):
            line = "unavailable"
        versions[tool] = line
    return versions


def preflight(staging_root: Path, store: S3ObjectStore | None = None) -> dict[str, Any]:
    usage = shutil.disk_usage(
        staging_root if staging_root.exists() else staging_root.parent
    )
    result: dict[str, Any] = {
        "git_sha": git_sha(),
        "staging_root": staging_root.as_posix(),
        "scratch_free_bytes": usage.free,
        "scratch_187gb": usage.free >= 187_000_000_000,
        "tools": tool_versions(),
    }
    if store is not None:
        store.probe()
        result["s3"] = "reachable"
        result["s3_root"] = store.config.root_uri
    return result


__all__ = ["config_hash", "git_sha", "preflight", "tool_versions"]
