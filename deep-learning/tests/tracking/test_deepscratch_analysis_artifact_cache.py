import os
from pathlib import Path
from types import SimpleNamespace

from exp.deepscratch.analysis.input import AnalysisRun, StudyAnalysisInput
from exp.deepscratch.identity import Variant


class _Client:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.downloads = 0

    def download_artifacts(self, run_id, artifact_path, destination):
        self.downloads += 1
        target = Path(destination) / Path(artifact_path).name
        target.write_bytes(self.payload + run_id.encode())
        return str(target)


def _run(run_id: str) -> AnalysisRun:
    return AnalysisRun(
        run_id=run_id,
        canonical_condition_id="condition",
        native_condition_id="condition",
        seed="1",
        variant=Variant.IMPLEMENTED,
        result=SimpleNamespace(artifact_aliases={}),
    )


def test_artifact_download_hides_mlflow_progress_and_restores_environment(
    tmp_path: Path, monkeypatch
) -> None:
    seen = []

    class Client(_Client):
        def download_artifacts(self, run_id, artifact_path, destination):
            seen.append(os.environ.get("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"))
            return super().download_artifacts(run_id, artifact_path, destination)

    monkeypatch.setenv("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "true")
    client = Client(b"payload-")
    data = StudyAnalysisInput(
        client, None, Variant.IMPLEMENTED, (),
        cache_dir=tmp_path / "artifacts", tracking_uri="sqlite:///one.db",
    )

    assert data.artifact_file(_run("run-1"), "metrics/history.csv") is not None
    assert seen == ["false"]
    assert os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] == "true"


def test_artifact_download_is_reused_for_the_same_mlflow_run(tmp_path: Path) -> None:
    cache = tmp_path / "shared-artifacts"
    first_client = _Client(b"first-")
    first = StudyAnalysisInput(
        first_client, None, Variant.IMPLEMENTED, (),
        cache_dir=cache, tracking_uri="sqlite:///one.db",
    )

    first_path = first.artifact_file(_run("run-1"), "metrics/history.csv")
    assert first_path is not None
    assert first_client.downloads == 1

    second_client = _Client(b"changed-analyzer-")
    second = StudyAnalysisInput(
        second_client, None, Variant.IMPLEMENTED, (),
        cache_dir=cache, tracking_uri="sqlite:///one.db",
    )
    assert second.artifact_file(_run("run-1"), "metrics/history.csv") == first_path
    assert second_client.downloads == 0

    changed_path = second.artifact_file(_run("run-2"), "metrics/history.csv")
    assert changed_path is not None
    assert changed_path != first_path
    assert second_client.downloads == 1


def test_artifact_cache_separates_mlflow_stores_with_equal_run_ids(tmp_path: Path) -> None:
    cache = tmp_path / "shared-artifacts"
    clients = (_Client(b"one-"), _Client(b"two-"))
    paths = []
    for uri, client in zip(("sqlite:///one.db", "sqlite:///two.db"), clients, strict=True):
        data = StudyAnalysisInput(
            client, None, Variant.IMPLEMENTED, (),
            cache_dir=cache, tracking_uri=uri,
        )
        paths.append(data.artifact_file(_run("same-run-id"), "artifact.csv"))

    assert paths[0] != paths[1]
    assert [client.downloads for client in clients] == [1, 1]
