"""Environment metadata, git inspection, and model parameter manifest extraction."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git(args: list[str], *, check: bool = True) -> str:
    result = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def current_git_info(entrypoint: str) -> dict[str, Any]:
    diff = _git(["diff"], check=False)
    return {
        "repository": _git(["rev-parse", "--show-toplevel"]).split("/")[-1],
        "commit": _git(["rev-parse", "HEAD"]),
        "branch": _git(["branch", "--show-current"]),
        "dirty": bool(_git(["status", "--porcelain"])),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "remote": _git(["remote", "get-url", "origin"], check=False),
        "entrypoint": entrypoint,
    }


def write_git_diff(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_git(["diff"], check=False), encoding="utf-8")


def environment_artifacts() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "system": platform.system().lower(),
        "kernel": platform.release(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def pip_freeze() -> str:
    return subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_manifest(model: Any) -> list[dict[str, Any]]:
    output = []
    for name, parameter in model.named_parameters():
        data = parameter.backend.to_numpy(parameter.data)
        output.append(
            {
                "name": name,
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "requires_grad": bool(parameter.requires_grad),
                "numel": int(data.size),
                "final_mean": float(data.mean()),
                "final_std": float(data.std()),
                "final_min": float(data.min()),
                "final_max": float(data.max()),
                "final_digest": hashlib.sha256(data.tobytes()).hexdigest(),
            }
        )
    return output


__all__ = [
    "current_git_info",
    "environment_artifacts",
    "file_digest",
    "parameter_manifest",
    "pip_freeze",
    "write_git_diff",
]
