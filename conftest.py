"""Repository-wide pytest configuration, RAMDisk acceleration, and state isolation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Redirect pytest temporary directories to in-memory /dev/shm (RAMDisk) if available."""
    if not config.option.basetemp:
        shm = Path("/dev/shm")
        if shm.is_dir() and os.access(shm, os.W_OK):
            uid = os.getuid() if hasattr(os, "getuid") else "tmp"
            basetemp = shm / f"pytest_repro_{uid}"
            basetemp.mkdir(parents=True, exist_ok=True)
            config.option.basetemp = str(basetemp)


@pytest.fixture(autouse=True)
def isolate_mlflow_default_artifact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure MLflow default tracking and artifacts always resolve to isolated in-memory tmp_path."""
    artifact_root = tmp_path / "mlflow-artifacts"
    monkeypatch.setenv("_MLFLOW_SERVER_ARTIFACT_ROOT", artifact_root.as_uri())
    monkeypatch.setenv(
        "F1_MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'tracking.db'}"
    )
