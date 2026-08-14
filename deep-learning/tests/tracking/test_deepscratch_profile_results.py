from pathlib import Path

from mlflow.tracking import MlflowClient

from exp.deepscratch.ds2.profile.result_writer import record_profile_result


def test_profile_result_is_marked_durable_after_artifact_upload(tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    output = tmp_path / "profile"
    output.mkdir()
    (output / "measurement.json").write_text('{"mean_ms": 1.5}\n', encoding="utf-8")

    run_id = record_profile_result(
        uri,
        volume="ds2",
        experiment_id="e02",
        variant="implemented",
        output=output,
    )

    client = MlflowClient(uri)
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["run.type"] == "profile"
    assert run.data.tags["result.schema.name"] == "ds2-profile"
    assert run.data.tags["result.durable_complete"] == "true"
    artifacts = {item.path for item in client.list_artifacts(run_id, "profile")}
    assert artifacts == {"profile/profile_manifest.json", "profile/raw"}
    assert (tmp_path / "artifacts").is_dir()
