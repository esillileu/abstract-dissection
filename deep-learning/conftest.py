"""Repository-wide pytest isolation for generated state."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_mlflow_default_artifact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep implicit SQLite MLflow artifacts out of the repository root.

    MLflow otherwise resolves an experiment without an explicit artifact
    location to ``./mlruns``. Tests deliberately use temporary SQLite tracking
    stores, so their matching default artifact store must be temporary too.
    """
    artifact_root = tmp_path / "mlflow-artifacts"
    monkeypatch.setenv("_MLFLOW_SERVER_ARTIFACT_ROOT", artifact_root.as_uri())
