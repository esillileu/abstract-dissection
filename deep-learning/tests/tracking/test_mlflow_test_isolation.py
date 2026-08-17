from pathlib import Path

from mlflow import MlflowClient


def test_sqlite_default_artifacts_are_isolated_from_repository(
    tmp_path: Path,
) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment_id = client.create_experiment("isolated")
    experiment = client.get_experiment(experiment_id)

    assert experiment.artifact_location == (
        tmp_path / "mlflow-artifacts" / experiment_id
    ).as_uri()
