import json
import os
from pathlib import Path

import pytest
from mlflow.entities import Dataset, DatasetInput, InputTag, Metric
from mlflow.tracking import MlflowClient

import mlprosection_mlflow.transfer as transfer
from mlprosection_mlflow.transfer import export_experiment, export_run, import_archive


def _tracking_uri(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.resolve() / 'mlflow.db'}"


def _artifact_uri(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve().as_uri()


def _source_run(
    client: MlflowClient, experiment_id: str, scratch_dir: Path
) -> str:
    run = client.create_run(
        experiment_id,
        start_time=1_700_000_000_000,
        run_name="test run",
        tags={"owner": "transfer-test", "metadata.unicode": "복원"},
    )
    run_id = run.info.run_id
    client.log_param(run_id, "optimizer", "adam")
    client.log_param(run_id, "epochs", "2")
    client.log_batch(
        run_id,
        metrics=[
            Metric("loss", 2.5, 1_700_000_001_000, 0),
            Metric("loss", 1.25, 1_700_000_002_000, 1),
            Metric("accuracy", 0.75, 1_700_000_002_000, 1),
        ],
    )
    client.log_inputs(
        run_id,
        datasets=[
            DatasetInput(
                Dataset(
                    name="test-dataset",
                    digest="sha256:test",
                    source_type="local",
                    source='{"path": "test.csv"}',
                    schema='{"columns": ["feature"]}',
                    profile='{"rows": 2}',
                ),
                [InputTag("context", "training")],
            )
        ],
    )
    scratch_dir.mkdir(parents=True, exist_ok=True)
    test_file = scratch_dir / "result.txt"
    test_file.write_text("portable artifact\n", encoding="utf-8")
    client.log_artifact(run_id, str(test_file), artifact_path="reports")
    client.set_terminated(
        run_id, status="FINISHED", end_time=1_700_000_012_345
    )
    return run_id


def _assert_restored(
    source_client: MlflowClient,
    source_run_id: str,
    target_client: MlflowClient,
    target_run_id: str,
) -> None:
    source = source_client.get_run(source_run_id)
    target = target_client.get_run(target_run_id)
    assert target.info.run_id != source.info.run_id
    assert target.info.start_time == source.info.start_time
    assert target.info.end_time == source.info.end_time
    assert (
        target.info.end_time - target.info.start_time
        == source.info.end_time - source.info.start_time
    )
    assert target.info.status == source.info.status
    assert target.info.run_name == source.info.run_name
    assert target.data.params == source.data.params
    for key, value in source.data.tags.items():
        assert target.data.tags[key] == value
    assert target.data.metrics == source.data.metrics
    assert target.inputs.dataset_inputs == source.inputs.dataset_inputs
    for key in source.data.metrics:
        source_history = source_client.get_metric_history(source_run_id, key)
        target_history = target_client.get_metric_history(target_run_id, key)
        assert [
            (item.value, item.timestamp, item.step) for item in target_history
        ] == [(item.value, item.timestamp, item.step) for item in source_history]
    restored_artifact = Path(
        target_client.download_artifacts(target_run_id, "reports/result.txt")
    )
    assert restored_artifact.read_text(encoding="utf-8") == "portable artifact\n"


def test_export_reports_all_phases_and_disables_mlflow_artifact_progress(
    tmp_path: Path, monkeypatch
) -> None:
    source_uri = _tracking_uri(tmp_path / "source")
    source_client = MlflowClient(tracking_uri=source_uri)
    experiment_id = source_client.create_experiment(
        "progress test",
        artifact_location=_artifact_uri(tmp_path / "source-artifacts"),
    )
    _source_run(source_client, experiment_id, tmp_path / "scratch")

    progress_calls = []

    class RecordingProgress:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.description = kwargs["desc"]
            self.updates = []
            progress_calls.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def __iter__(self):
            return iter(self.iterable)

        def update(self, amount):
            self.updates.append(amount)

    monkeypatch.setattr(transfer, "tqdm", RecordingProgress)
    monkeypatch.delenv("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", raising=False)
    original_download = transfer._download_artifacts
    artifact_progress_values = []

    def checked_download(client, run_id, destination):
        artifact_progress_values.append(
            os.environ.get("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR")
        )
        original_download(client, run_id, destination)

    monkeypatch.setattr(transfer, "_download_artifacts", checked_download)

    archive = export_experiment(
        source_uri, "progress test", tmp_path / "progress.zip"
    )

    assert archive.is_file()
    assert [call.description for call in progress_calls] == [
        "Finding finished runs",
        "Collecting run data",
        "Downloading artifacts",
        "Creating archive",
    ]
    assert progress_calls[0].updates == [1]
    assert artifact_progress_values == ["false"]
    assert "MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR" not in os.environ


def test_export_import_single_run_restores_complete_run(
    tmp_path: Path, monkeypatch
) -> None:
    detected_tags = {
        "transfer.destination.hostname": "import-server",
        "transfer.destination.cpu.model": "Test CPU",
        "transfer.destination.cpu.logical_count": "16",
        "transfer.destination.memory.total_bytes": "68719476736",
        "transfer.destination.gpu.count": "1",
        "transfer.destination.gpu.0.name": "Test GPU",
    }
    monkeypatch.setattr(transfer, "_environment_tags", lambda _: detected_tags)
    source_client = MlflowClient(tracking_uri=_tracking_uri(tmp_path / "source"))
    experiment_id = source_client.create_experiment(
        "test experiment",
        artifact_location=_artifact_uri(tmp_path / "source-artifacts"),
        tags={"purpose": "transfer verification", "shared": "source"},
    )
    source_run_id = _source_run(source_client, experiment_id, tmp_path / "scratch")

    archive = export_run(
        _tracking_uri(tmp_path / "source"),
        source_run_id,
        tmp_path / "run.zip",
    )
    target_uri = _tracking_uri(tmp_path / "target")
    target_client = MlflowClient(tracking_uri=target_uri)
    target_experiment_id = target_client.create_experiment(
        "imported test experiment",
        artifact_location=_artifact_uri(tmp_path / "target-artifacts"),
        tags={"target.only": "kept", "shared": "target"},
    )
    with pytest.raises(ValueError, match="--reuse-experiment"):
        import_archive(
            target_uri,
            archive,
            experiment_name="imported test experiment",
            capture_environment=False,
        )

    result = import_archive(
        target_uri,
        archive,
        experiment_name="imported test experiment",
        destination_tags={"server.name": "training-box-a"},
        reuse_experiment=True,
    )

    target_experiment = target_client.get_experiment(result["experiment_id"])
    assert result["experiment_id"] == target_experiment_id
    assert result["reused_experiment"] is True
    assert target_experiment.tags == {
        "purpose": "transfer verification",
        "shared": "target",
        "target.only": "kept",
        **detected_tags,
        "server.name": "training-box-a",
    }
    _assert_restored(
        source_client,
        source_run_id,
        target_client,
        result["run_id_map"][source_run_id],
    )
    imported_run = target_client.get_run(result["run_id_map"][source_run_id])
    for key, value in {**detected_tags, "server.name": "training-box-a"}.items():
        assert imported_run.data.tags[key] == value


def test_export_import_experiment_remaps_nested_run_ids(tmp_path: Path) -> None:
    source_uri = _tracking_uri(tmp_path / "source")
    target_uri = _tracking_uri(tmp_path / "target")
    source_client = MlflowClient(tracking_uri=source_uri)
    experiment_id = source_client.create_experiment(
        "nested test experiment",
        artifact_location=_artifact_uri(tmp_path / "source-artifacts"),
    )
    parent_id = _source_run(source_client, experiment_id, tmp_path / "scratch")
    child = source_client.create_run(
        experiment_id,
        start_time=1_700_000_020_000,
        run_name="child",
        tags={
            "mlflow.parentRunId": parent_id,
            "parent.mlflow_run_id": parent_id,
        },
    )
    source_client.log_metric(
        child.info.run_id, "child_metric", 42.0, timestamp=1_700_000_021_000, step=3
    )
    source_client.set_terminated(
        child.info.run_id, status="FINISHED", end_time=1_700_000_022_000
    )
    unfinished = source_client.create_run(
        experiment_id,
        start_time=1_700_000_030_000,
        run_name="still running",
    )

    archive = export_experiment(
        source_uri, "nested test experiment", tmp_path / "experiment.zip"
    )
    result = import_archive(
        target_uri,
        archive,
        artifact_location=_artifact_uri(tmp_path / "target-artifacts"),
    )
    assert result["reused_experiment"] is False

    target_client = MlflowClient(tracking_uri=target_uri)
    new_parent_id = result["run_id_map"][parent_id]
    new_child_id = result["run_id_map"][child.info.run_id]
    assert unfinished.info.run_id not in result["run_id_map"]
    imported_child = target_client.get_run(new_child_id)
    assert new_parent_id != parent_id
    assert imported_child.data.tags["mlflow.parentRunId"] == new_parent_id
    assert imported_child.data.tags["parent.mlflow_run_id"] == new_parent_id
    history = target_client.get_metric_history(new_child_id, "child_metric")
    assert [(item.value, item.timestamp, item.step) for item in history] == [
        (42.0, 1_700_000_021_000, 3)
    ]


def test_export_run_includes_non_finished_parent_dependency(
    tmp_path: Path,
) -> None:
    source_uri = _tracking_uri(tmp_path / "source")
    source_client = MlflowClient(tracking_uri=source_uri)
    experiment_id = source_client.create_experiment(
        "single child source",
        artifact_location=_artifact_uri(tmp_path / "source-artifacts"),
    )
    parent = source_client.create_run(
        experiment_id,
        tags={
            "run.type": "condition_parent",
            "condition.group.key": "group",
        },
    )
    child = source_client.create_run(
        experiment_id,
        tags={
            "run.type": "seed_trial",
            "run.key": "seed",
            "condition.group.key": "group",
            "mlflow.parentRunId": parent.info.run_id,
            "parent.mlflow_run_id": parent.info.run_id,
        },
    )
    source_client.set_terminated(child.info.run_id)

    archive = export_run(source_uri, child.info.run_id, tmp_path / "child.zip")
    result = import_archive(
        _tracking_uri(tmp_path / "target"),
        archive,
        capture_environment=False,
    )

    assert set(result["run_id_map"]) == {
        parent.info.run_id,
        child.info.run_id,
    }
    target_client = MlflowClient(tracking_uri=_tracking_uri(tmp_path / "target"))
    imported_child = target_client.get_run(result["run_id_map"][child.info.run_id])
    assert imported_child.data.tags["mlflow.parentRunId"] == result["run_id_map"][
        parent.info.run_id
    ]


def test_importing_parent_relinks_existing_children_in_reused_experiment(
    tmp_path: Path,
) -> None:
    source_uri = _tracking_uri(tmp_path / "source")
    source_client = MlflowClient(tracking_uri=source_uri)
    source_experiment_id = source_client.create_experiment("source")
    source_parent = source_client.create_run(
        source_experiment_id,
        tags={
            "run.type": "condition_parent",
            "condition.group.key": "group",
        },
    )
    source_client.set_terminated(source_parent.info.run_id)
    archive = export_run(
        source_uri,
        source_parent.info.run_id,
        tmp_path / "parent.zip",
    )

    target_uri = _tracking_uri(tmp_path / "target")
    target_client = MlflowClient(tracking_uri=target_uri)
    target_experiment_id = target_client.create_experiment("target")
    child = target_client.create_run(
        target_experiment_id,
        tags={
            "run.type": "seed_trial",
            "run.key": "seed",
            "condition.group.key": "group",
            "mlflow.parentRunId": "source-only-parent",
            "parent.mlflow_run_id": "source-only-parent",
        },
    )

    result = import_archive(
        target_uri,
        archive,
        experiment_name="target",
        capture_environment=False,
        reuse_experiment=True,
    )

    imported_parent_id = result["run_id_map"][source_parent.info.run_id]
    tags = target_client.get_run(child.info.run_id).data.tags
    assert tags["mlflow.parentRunId"] == imported_parent_id
    assert tags["parent.mlflow_run_id"] == imported_parent_id
    assert result["relationship_repairs"][0]["run_id"] == child.info.run_id


def test_import_restores_reused_target_when_source_run_is_active(
    tmp_path: Path,
) -> None:
    source_uri = _tracking_uri(tmp_path / "source")
    source_client = MlflowClient(tracking_uri=source_uri)
    source_experiment_id = source_client.create_experiment("source")
    source_parent = source_client.create_run(
        source_experiment_id,
        tags={
            "run.type": "condition_parent",
            "condition.group.key": "group",
        },
    )
    source_client.set_terminated(source_parent.info.run_id)
    archive = export_run(
        source_uri,
        source_parent.info.run_id,
        tmp_path / "parent.zip",
    )

    target_uri = _tracking_uri(tmp_path / "target")
    target_client = MlflowClient(tracking_uri=target_uri)
    target_experiment_id = target_client.create_experiment("target")
    deleted_parent = target_client.create_run(
        target_experiment_id,
        tags={
            "run.type": "condition_parent",
            "condition.group.key": "group",
        },
    )
    target_client.delete_run(deleted_parent.info.run_id)

    result = import_archive(
        target_uri,
        archive,
        experiment_name="target",
        capture_environment=False,
        reuse_experiment=True,
    )

    assert result["run_id_map"][source_parent.info.run_id] == (
        deleted_parent.info.run_id
    )
    assert target_client.get_run(
        deleted_parent.info.run_id
    ).info.lifecycle_stage == "active"


def test_importing_same_archive_twice_reuses_run(tmp_path: Path) -> None:
    source_uri = _tracking_uri(tmp_path / "source")
    source_client = MlflowClient(tracking_uri=source_uri)
    experiment_id = source_client.create_experiment(
        "idempotent source",
        artifact_location=_artifact_uri(tmp_path / "source-artifacts"),
    )
    source_run_id = _source_run(source_client, experiment_id, tmp_path / "scratch")
    archive = export_run(source_uri, source_run_id, tmp_path / "run.zip")
    target_uri = _tracking_uri(tmp_path / "target")

    first = import_archive(
        target_uri,
        archive,
        experiment_name="idempotent target",
        artifact_location=_artifact_uri(tmp_path / "target-artifacts"),
        capture_environment=False,
    )
    second = import_archive(
        target_uri,
        archive,
        experiment_name="idempotent target",
        capture_environment=False,
        reuse_experiment=True,
    )

    assert second["run_id_map"] == first["run_id_map"]
    client = MlflowClient(tracking_uri=target_uri)
    assert len(client.search_runs([first["experiment_id"]])) == 1


def test_export_includes_local_only_checkpoint_and_import_verifies_it(
    tmp_path: Path,
) -> None:
    source_uri = _tracking_uri(tmp_path / "source")
    source_client = MlflowClient(tracking_uri=source_uri)
    experiment_id = source_client.create_experiment(
        "checkpoint source",
        artifact_location=_artifact_uri(tmp_path / "source-artifacts"),
    )
    run = source_client.create_run(
        experiment_id,
        tags={
            "run.type": "seed_trial",
            "run.key": "seed-key",
            "execution_group.id": "GT07",
        },
    )
    checkpoint = tmp_path / "local-checkpoints" / "generations" / "latest-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "weights.bin").write_bytes(b"checkpoint")
    digest = transfer._path_digest(checkpoint)
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    manifest = manifest_dir / "checkpoint_manifest.json"
    manifest.write_text(json.dumps({
        "format": "v2",
        "local_root": str(checkpoint.parent.parent),
        "latest": {"path": str(checkpoint), "digest": digest},
        "best": None,
    }), encoding="utf-8")
    source_client.log_artifact(
        run.info.run_id, str(manifest), artifact_path="checkpoints"
    )
    source_client.set_terminated(run.info.run_id)

    archive = export_run(source_uri, run.info.run_id, tmp_path / "checkpoint.zip")
    target_uri = _tracking_uri(tmp_path / "target")
    result = import_archive(
        target_uri,
        archive,
        experiment_name="checkpoint target",
        artifact_location=_artifact_uri(tmp_path / "target-artifacts"),
        capture_environment=False,
    )

    imported = MlflowClient(tracking_uri=target_uri).get_run(
        result["run_id_map"][run.info.run_id]
    )
    assert imported.data.tags["checkpoint.latest.status"] == "present"
    assert imported.data.tags["checkpoint.latest.sha256"] == digest
    assert imported.data.tags["checkpoint.best.status"] == "missing"
