from __future__ import annotations

from mlprosection_mlflow.schema_v1 import SchemaV1Run, _save_checkpoint


class _CheckpointModel:
    def save_params_npz(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")


def test_yaml_contract_projects_to_the_schema_v1_artifact_tree(tmp_path, monkeypatch) -> None:
    import mlprosection_mlflow.schema_v1 as schema

    monkeypatch.setattr(schema, "ARTIFACT_ROOT", tmp_path)
    run = SchemaV1Run({
        "kind": "optimizer_toy", "seed": 7, "atomic_run_id": "TOY-SGD",
        "experiment_ids": ["e01"], "execution_group_id": "g01", "recipe_id": "RC-TOY-OPT",
        "structure_signature": "analytic-toy-v1", "dataset": {"id": "DS-TOY"},
        "model": {"family": "analytic", "task_type": "optimization"},
        "training": {"entrypoint": "experiments/configs/toy.yaml", "max_epochs": 0},
    })

    run.write_artifacts(
        model=None,
        final_metrics={"final/status/success": 1.0},
        history_rows=[
            ("update", 20, "train/loss", 1.5),
            ("eval", 1, "valid/loss", 1.25),
            ("epoch", 1, "train/loss", 1.0),
        ],
        profiling_metrics={},
    )

    assert run.identity.schema_version == 1
    assert run.artifact_root == tmp_path / "mlprosection" / "results" / "mlflow_artifacts" / run.identity.run_key
    assert run.local_checkpoint_root == tmp_path / "mlprosection" / "results" / "checkpoints" / run.identity.run_key
    assert (run.artifact_root / "config/resolved.json").is_file()
    assert (run.artifact_root / "metrics/history.csv").is_file()
    assert "update,20,train/loss,1.5" in (run.artifact_root / "metrics/update_history.csv").read_text()
    assert "eval,1,valid/loss,1.25" in (run.artifact_root / "metrics/eval_history.csv").read_text()
    assert "epoch,1,train/loss,1.0" in (run.artifact_root / "metrics/epoch_history.csv").read_text()
    assert (run.artifact_root / "metrics/final.json").is_file()
    assert (run.artifact_root / "checkpoints/checkpoint_manifest.json").is_file()


def test_final_checkpoint_can_be_disabled(tmp_path) -> None:
    path, digest = _save_checkpoint(tmp_path, _CheckpointModel(), save_final=False)

    assert path is None
    assert digest is None


def test_final_checkpoint_is_saved_by_default(tmp_path) -> None:
    path, digest = _save_checkpoint(tmp_path, _CheckpointModel(), save_final=True)

    assert path == tmp_path / "final.npz"
    assert digest is not None
