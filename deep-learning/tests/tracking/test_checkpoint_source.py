from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mlprosection_mlflow.checkpoint_source import resolve_checkpoint_source


class _Client:
    def __init__(
        self,
        tmp_path: Path,
        *,
        payload: bool = True,
        legacy_path: bool = False,
    ) -> None:
        self.tmp_path = tmp_path
        self.payload = payload
        self.legacy_path = legacy_path
        self.requested_artifacts = []
        self.run = SimpleNamespace(
            info=SimpleNamespace(run_id="source-run"),
            data=SimpleNamespace(
                params={"seed/master": "7"},
                tags={"run.key": "remote-run-key"},
            ),
        )

    def get_experiment_by_name(self, name):
        assert name == "ds2"
        return SimpleNamespace(experiment_id="10")

    def search_runs(self, **kwargs):
        assert "GT07" in kwargs["filter_string"]
        assert "SEQD-ATTN-REV" in kwargs["filter_string"]
        return [self.run]

    def download_artifacts(self, run_id, artifact_path, destination=None):
        assert run_id == "source-run"
        self.requested_artifacts.append(artifact_path)
        if artifact_path == "checkpoints.csv":
            index = self.tmp_path / "checkpoints.csv"
            index.write_text(
                "update,epoch,kind,path\n"
                "10,1,selected,/source/best-epoch-0001\n",
                encoding="utf-8",
            )
            return str(index)
        if not self.payload:
            raise RuntimeError("missing")
        if (
            self.legacy_path
            and artifact_path == "checkpoints/generations/best-epoch-0001"
        ):
            raise RuntimeError("missing generation path")
        assert artifact_path in {
            "checkpoints/generations/best-epoch-0001",
            "checkpoints/best-epoch-0001",
        }
        checkpoint = Path(destination) / "best-epoch-0001"
        checkpoint.mkdir(parents=True)
        (checkpoint / "manifest.json").write_text(
            '{"schema_version": 2}',
            encoding="utf-8",
        )
        return str(checkpoint)


def _config() -> dict[str, object]:
    return {
        "seed": 7,
        "tracking": {"enabled": True, "experiment": "ds2"},
        "checkpoint": {
            "source_group_id": "GT07",
            "source_atomic_run_id": "SEQD-ATTN-REV",
            "source_kind": "selected",
            "source_path": None,
        },
    }


def test_matching_seed_selected_checkpoint_is_downloaded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _config()

    client = _Client(tmp_path)
    resolved = resolve_checkpoint_source(config, client=client)

    assert resolved == (
        tmp_path
        / "exp/ds2/results/source_checkpoints/source-run/best-epoch-0001"
    ).relative_to(tmp_path)
    assert config["checkpoint"]["source_path"] == str(resolved)
    assert client.requested_artifacts[-1] == (
        "checkpoints/generations/best-epoch-0001"
    )


def test_missing_selected_payload_has_an_actionable_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="upload_eval_checkpoints=true"):
        resolve_checkpoint_source(
            _config(),
            client=_Client(tmp_path, payload=False),
        )


def test_legacy_checkpoint_artifact_path_remains_supported(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    client = _Client(tmp_path, legacy_path=True)

    resolved = resolve_checkpoint_source(_config(), client=client)

    assert resolved.is_dir()
    assert client.requested_artifacts[-2:] == [
        "checkpoints/generations/best-epoch-0001",
        "checkpoints/best-epoch-0001",
    ]
