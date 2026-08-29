"""DLFS pytest isolation for generated MLflow artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_mlflow_default_artifact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep SQLite-backed test artifacts out of the repository ``mlruns`` root."""
    artifact_root = tmp_path / "mlflow-artifacts"
    monkeypatch.setenv("_MLFLOW_SERVER_ARTIFACT_ROOT", artifact_root.as_uri())
