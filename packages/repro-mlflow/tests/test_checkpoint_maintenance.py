from __future__ import annotations

import json
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from repro_mlflow.checkpoint_maintenance import (
    checkpoint_backfill,
    checkpoint_prune,
    dedupe,
    path_digest,
    resolve_local_roles,
)


def _tracking_uri(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.resolve() / 'mlflow.db'}"


def _artifact_uri(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve().as_uri()


def _checkpoint(root: Path, role: str, content: bytes) -> Path:
    generation = root / "generations" / f"{role}-1"
    generation.mkdir(parents=True)
    (generation / "weights.bin").write_bytes(content)
    digest = path_digest(generation)
    (root / f"{role}.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "role": role,
                "path": f"generations/{generation.name}",
                "sha256": digest,
                "epoch": 1,
                "update": 2,
            }
        ),
        encoding="utf-8",
    )
    return generation


def test_resolve_local_roles_supports_v2_and_legacy_final(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    latest = _checkpoint(root, "latest", b"latest")
    (root / "final.npz").write_bytes(b"legacy")

    roles = resolve_local_roles(root)

    assert roles["latest"]["path"] == latest
    assert roles["latest"]["source"] == "v2"
    (root / "latest.json").unlink()
    roles = resolve_local_roles(root)
    assert roles["latest"]["path"] == root / "final.npz"
    assert roles["latest"]["source"] == "legacy-final"


def test_resolve_local_roles_rejects_digest_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    _checkpoint(root, "latest", b"latest")
    pointer = json.loads((root / "latest.json").read_text())
    pointer["sha256"] = "wrong"
    (root / "latest.json").write_text(json.dumps(pointer))

    with pytest.raises(ValueError, match="digest mismatch"):
        resolve_local_roles(root)


def test_backfill_is_dry_run_by_default_and_prune_keeps_only_roles(
    tmp_path: Path,
) -> None:
    tracking_uri = _tracking_uri(tmp_path / "tracking")
    artifact_root = tmp_path / "artifacts"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment(
        "ds1", artifact_location=_artifact_uri(artifact_root / "1")
    )
    run = client.create_run(
        experiment_id,
        tags={"run.type": "seed_trial", "run.key": "seed-key"},
    )
    client.set_terminated(run.info.run_id)
    local_root = tmp_path / "exp/ds1/results/checkpoints/seed-key"
    latest = _checkpoint(local_root, "latest", b"latest")
    stale = local_root / "generations/periodic-1"
    stale.mkdir()
    (stale / "weights.bin").write_bytes(b"stale")

    dry_run = checkpoint_backfill(tracking_uri, ["ds1"], repository_root=tmp_path)
    assert any(item["action"] == "upload" for item in dry_run["entries"])
    assert client.list_artifacts(run.info.run_id, "checkpoints") == []

    checkpoint_backfill(tracking_uri, ["ds1"], apply=True, repository_root=tmp_path)
    names = {
        item.path for item in client.list_artifacts(run.info.run_id, "checkpoints")
    }
    assert "checkpoints/latest.json" in names
    assert (
        client.get_run(run.info.run_id).data.tags["checkpoint.best.status"] == "missing"
    )

    prune_dry_run = checkpoint_prune(
        tracking_uri,
        ["ds1"],
        repository_root=tmp_path,
        artifact_root=artifact_root,
    )
    assert stale.exists()
    assert any(item["path"] == str(stale) for item in prune_dry_run["entries"])
    checkpoint_prune(
        tracking_uri,
        ["ds1"],
        apply=True,
        repository_root=tmp_path,
        artifact_root=artifact_root,
    )
    assert latest.exists()
    assert not stale.exists()


def test_dedupe_prefers_completeness_over_recency_and_reparents(
    tmp_path: Path,
) -> None:
    tracking_uri = _tracking_uri(tmp_path / "tracking")
    artifact_root = tmp_path / "artifacts"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment(
        "ds2", artifact_location=_artifact_uri(artifact_root / "1")
    )
    old_parent = client.create_run(
        experiment_id,
        start_time=1,
        tags={"run.type": "condition_parent", "condition.group.key": "group"},
    )
    new_parent = client.create_run(
        experiment_id,
        start_time=2,
        tags={"run.type": "condition_parent", "condition.group.key": "group"},
    )
    complete = client.create_run(
        experiment_id,
        start_time=1,
        tags={
            "run.type": "seed_trial",
            "run.key": "duplicate",
            "condition.group.key": "group",
            "mlflow.parentRunId": old_parent.info.run_id,
        },
    )
    incomplete = client.create_run(
        experiment_id,
        start_time=2,
        tags={
            "run.type": "seed_trial",
            "run.key": "duplicate",
            "condition.group.key": "group",
            "mlflow.parentRunId": old_parent.info.run_id,
        },
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    for name in ("updates.csv", "evaluations.csv"):
        path = scratch / name
        path.write_text("header\n")
        client.log_artifact(complete.info.run_id, str(path))
    for run_id in (
        old_parent.info.run_id,
        new_parent.info.run_id,
        complete.info.run_id,
        incomplete.info.run_id,
    ):
        client.set_terminated(run_id)

    report = dedupe(tracking_uri, "ds2", apply=True, artifact_root=artifact_root)

    seed_entry = next(
        item for item in report["entries"] if item["run_type"] == "seed_trial"
    )
    assert seed_entry["winner_run_id"] == complete.info.run_id
    assert client.get_run(incomplete.info.run_id).info.lifecycle_stage == "deleted"
    canonical_parent = new_parent.info.run_id
    assert (
        client.get_run(complete.info.run_id).data.tags["mlflow.parentRunId"]
        == canonical_parent
    )
