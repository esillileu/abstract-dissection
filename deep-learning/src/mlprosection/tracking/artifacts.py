from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    """Write formatted JSON, creating parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    """Write UTF-8 text, creating parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def write_history_csv(
    path: Path,
    *,
    run_key: str,
    rows: list[tuple[str, int, str, float]],
) -> None:
    """Write long-format metric history CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.time()
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["run_key", "step_type", "step", "metric", "value", "timestamp"])
        for step_type, step, metric, value in rows:
            writer.writerow([run_key, step_type, step, metric, value, timestamp])


def write_runtime_history_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write epoch-level runtime CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step_type",
        "step",
        "train_s",
        "eval_s",
        "checkpoint_s",
        "throughput_samples_per_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_memory_history_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write sampled memory CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp_s",
        "cpu_rss_bytes",
        "gpu_used_bytes",
        "gpu_reserved_bytes",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def current_git_info(entrypoint: str) -> dict[str, Any]:
    """Collect git state for schema v1 code artifacts."""

    return {
        "repository": _git(["rev-parse", "--show-toplevel"]).split("/")[-1],
        "commit": _git(["rev-parse", "HEAD"]),
        "branch": _git(["branch", "--show-current"]),
        "dirty": bool(_git(["status", "--porcelain"])),
        "remote": _git(["remote", "get-url", "origin"], check=False),
        "entrypoint": entrypoint,
    }


def write_git_diff(path: Path) -> None:
    """Write current git diff patch."""

    diff = _git(["diff"], check=False)
    write_text(path, diff)


def environment_artifacts() -> dict[str, Any]:
    """Return minimal environment metadata."""

    return {
        "platform": platform.platform(),
        "system": platform.system().lower(),
        "kernel": platform.release(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def pip_freeze() -> str:
    """Return installed packages from the running interpreter."""

    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout


def file_digest(path: Path) -> str:
    """Return a SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_manifest(model) -> list[dict[str, Any]]:
    """Create initial/final parameter statistics from named_parameters()."""

    manifest = []
    for name, parameter in model.named_parameters():
        data = parameter.backend.to_numpy(parameter.data)
        manifest.append(
            {
                "name": name,
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "requires_grad": bool(parameter.requires_grad),
                "numel": int(data.size),
                "initial_mean": float(data.mean()),
                "initial_std": float(data.std()),
                "initial_min": float(data.min()),
                "initial_max": float(data.max()),
                "initial_norm": float((data * data).sum() ** 0.5),
                "initial_digest": hashlib.sha256(data.tobytes()).hexdigest(),
            }
        )
    return manifest


def _git(args: list[str], *, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()
