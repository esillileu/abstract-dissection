from pathlib import Path

from mlflow.tracking import MlflowClient

from mlprosection_mlflow.run_relationships import relink_parents


def _tracking_uri(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.resolve() / 'mlflow.db'}"


def test_relink_parents_is_dry_run_by_default_and_updates_both_tags(
    tmp_path: Path,
) -> None:
    tracking_uri = _tracking_uri(tmp_path)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment("ds2")
    parent = client.create_run(
        experiment_id,
        tags={
            "run.type": "condition_parent",
            "condition.group.key": "group",
        },
    )
    child = client.create_run(
        experiment_id,
        tags={
            "run.type": "seed_trial",
            "condition.group.key": "group",
            "mlflow.parentRunId": "source-parent",
            "parent.mlflow_run_id": "source-parent",
        },
    )

    dry_run = relink_parents(tracking_uri, ["ds2"])

    assert dry_run["mode"] == "dry-run"
    assert dry_run["entries"][0]["action"] == "relink"
    assert client.get_run(child.info.run_id).data.tags[
        "mlflow.parentRunId"
    ] == "source-parent"

    applied = relink_parents(tracking_uri, ["ds2"], apply=True)

    assert applied["mode"] == "apply"
    tags = client.get_run(child.info.run_id).data.tags
    assert tags["mlflow.parentRunId"] == parent.info.run_id
    assert tags["parent.mlflow_run_id"] == parent.info.run_id


def test_relink_parents_does_not_guess_between_duplicate_parents(
    tmp_path: Path,
) -> None:
    tracking_uri = _tracking_uri(tmp_path)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment("ds2")
    for _ in range(2):
        client.create_run(
            experiment_id,
            tags={
                "run.type": "condition_parent",
                "condition.group.key": "group",
            },
        )
    child = client.create_run(
        experiment_id,
        tags={
            "run.type": "seed_trial",
            "condition.group.key": "group",
            "mlflow.parentRunId": "source-parent",
        },
    )

    report = relink_parents(tracking_uri, ["ds2"], apply=True)

    assert report["entries"][0]["action"] == "ambiguous"
    assert client.get_run(child.info.run_id).data.tags[
        "mlflow.parentRunId"
    ] == "source-parent"


def test_relink_parents_restores_uniquely_referenced_deleted_parent(
    tmp_path: Path,
) -> None:
    tracking_uri = _tracking_uri(tmp_path)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment("ds2")
    parent = client.create_run(
        experiment_id,
        tags={
            "run.type": "condition_parent",
            "condition.group.key": "group",
        },
    )
    child = client.create_run(
        experiment_id,
        tags={
            "run.type": "seed_trial",
            "condition.group.key": "group",
            "mlflow.parentRunId": parent.info.run_id,
            "parent.mlflow_run_id": parent.info.run_id,
        },
    )
    client.delete_run(parent.info.run_id)

    report = relink_parents(tracking_uri, ["ds2"], apply=True)

    assert report["entries"] == [{
        "action": "restore-parent",
        "experiment_id": experiment_id,
        "condition_group_key": "group",
        "parent_run_id": parent.info.run_id,
        "affected_child_run_ids": [child.info.run_id],
        "reason": "the uniquely referenced condition parent is soft-deleted",
    }]
    assert client.get_run(parent.info.run_id).info.lifecycle_stage == "active"
