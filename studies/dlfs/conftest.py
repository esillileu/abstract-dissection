"""DLFS pytest isolation for generated MLflow artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def _mlflow_sqlite_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Pre-initialize a single SQLite MLflow schema once per session."""
    template_dir = tmp_path_factory.mktemp("mlflow_template")
    db_path = template_dir / "template.db"
    from mlflow.tracking import MlflowClient

    client = MlflowClient(f"sqlite:///{db_path}")
    exp_id = client.create_experiment("init")
    client.delete_experiment(exp_id)
    return db_path


@pytest.fixture(autouse=True)
def isolate_mlflow_default_artifact_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _mlflow_sqlite_template: Path,
) -> None:
    """Keep SQLite-backed test artifacts out of the repository ``mlruns`` root."""
    artifact_root = tmp_path / "mlflow-artifacts"
    monkeypatch.setenv("_MLFLOW_SERVER_ARTIFACT_ROOT", artifact_root.as_uri())
    tracking_db = tmp_path / "tracking.db"
    mlflow_db = tmp_path / "mlflow.db"
    shutil.copyfile(_mlflow_sqlite_template, tracking_db)
    shutil.copyfile(_mlflow_sqlite_template, mlflow_db)
    monkeypatch.setenv("F1_MLFLOW_TRACKING_URI", f"sqlite:///{tracking_db}")
