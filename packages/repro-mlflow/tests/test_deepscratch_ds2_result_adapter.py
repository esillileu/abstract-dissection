from pathlib import Path

from mlflow.exceptions import MlflowException

from dlfs.ds2.original.result_adapter import (
    _word2vec_checkpoint_projection,
)


def test_missing_legacy_checkpoint_does_not_abort_analysis(
    tmp_path: Path, monkeypatch
) -> None:
    class MissingArtifactCache:
        def get(self, run_id: str, artifact_path: str) -> Path:
            raise MlflowException("artifact does not exist")

    monkeypatch.setenv("EXP_CACHE_ROOT", str(tmp_path / "cache"))

    assert (
        _word2vec_checkpoint_projection(
            object(), "run-without-checkpoint", artifact_cache=MissingArtifactCache()
        )
        is None
    )
