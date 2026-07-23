from __future__ import annotations

from mlprosection_mlflow.schema_v1 import SchemaV1Run, _select_final_checkpoint


def test_yaml_contract_projects_to_the_schema_v1_artifact_tree(tmp_path, monkeypatch) -> None:
    import mlprosection_mlflow.schema_v1 as schema

    monkeypatch.setattr(schema, "ARTIFACT_ROOT", tmp_path)
    run = SchemaV1Run({
        "kind": "supervised_classification", "seed": 7, "atomic_run_id": "MLP-OPT-SGD",
        "experiment_ids": ["e01"], "execution_group_id": "g01", "recipe_id": "RC-TOY-OPT",
        "structure_signature": "analytic-toy-v1", "dataset": {"id": "DS-TOY"},
        "model": {"family": "analytic", "task_type": "optimization"},
        "training": {"entrypoint": "exp/ds1/config/e01_mnist_optimizer.yaml", "max_epochs": 0},
    })

    run.write_artifacts(
        model=None,
        final_metrics={"final/status/success": 1.0},
        metric_rows=[
            (20, "update/train/loss", 1.5),
            (1, "update/eval_valid/loss", 1.25),
            (1, "epoch/train/loss", 1.0),
        ],
        profiling_metrics={},
    )

    assert run.identity.schema_version == 1
    assert run.artifact_root == tmp_path / "mlprosection" / "results" / "mlflow_artifacts" / run.identity.run_key
    assert run.local_checkpoint_root == tmp_path / "mlprosection" / "results" / "checkpoints" / run.identity.run_key
    assert (run.artifact_root / "config/resolved.json").is_file()
    assert (run.artifact_root / "metrics/metrics.csv").is_file()
    metrics_text = (run.artifact_root / "metrics/metrics.csv").read_text()
    assert "20,update/train/loss,1.5" in metrics_text
    assert "1,update/eval_valid/loss,1.25" in metrics_text
    assert "1,epoch/train/loss,1.0" in metrics_text
    assert (run.artifact_root / "metrics/final.json").is_file()
    assert (run.artifact_root / "checkpoints/checkpoint_manifest.json").is_file()


def test_final_checkpoint_can_be_disabled(tmp_path) -> None:
    path, digest = _select_final_checkpoint(tmp_path, save_final=False)

    assert path is None
    assert digest is None


def test_final_checkpoint_selects_latest_v2_generation(tmp_path) -> None:
    generation = tmp_path / "generations" / "latest-1"
    generation.mkdir(parents=True)
    (tmp_path / "latest.json").write_text(
        '{"schema_version": 2, "path": "generations/latest-1", "sha256": "abc"}'
    )
    path, digest = _select_final_checkpoint(tmp_path, save_final=True)

    assert path == generation
    assert digest == "abc"
