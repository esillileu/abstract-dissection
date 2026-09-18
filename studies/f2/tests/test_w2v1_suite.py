from __future__ import annotations

import hashlib
from contextlib import contextmanager

import numpy as np

from f2.definition import DEFINITION
from f2.suites.w2v.artifacts import load_checkpoint, save_checkpoint
from f2.suites.w2v1.executor import create_session, restore_session
from f2.suites.w2v1.tracked import run_tracked_yaml
from f2.suites.w2v1.validation import analyze_latest, check_latest
from repro_core.context import ExperimentContext, RuntimePaths
from repro_core.execution.runner import run_config


def _paths(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    return RuntimePaths(
        repo_root=DEFINITION.get_suite("w2v1").config_root.parents[6],
        data_root=root / "data",
        artifacts_root=root / "artifacts",
        cache_root=root / "cache",
        staging_root=root / "staging",
        references_root=root / "references",
        studies_root=root / "studies",
    )


def test_w2v1_local_vertical_slice_and_resume_parity(tmp_path):
    definition = DEFINITION.get_suite("w2v1")
    assert definition.executor_module == "f2.suites.w2v1.executor"
    source = definition.config_root / "e01_table2_cbow.yaml"
    spec = definition.load_run_spec(source, atomic_run_id="local-smoke", overrides={})
    config = spec.to_executor_config()
    paths = _paths(tmp_path)

    result = run_config(
        config,
        ExperimentContext(paths=paths),
        executor_module=definition.executor_module,
    )
    final_state = load_checkpoint(result.checkpoint)
    assert final_state.completed_epochs == 2
    assert result.lookup.is_dir()
    assert result.metrics.stat().st_size > 0

    interrupted = create_session(config, paths.repo_root)
    interrupted.train_epoch()
    interrupted_checkpoint = tmp_path / "interrupted"
    save_checkpoint(
        interrupted.export_state(),
        interrupted_checkpoint,
        resource_version="w2v1-local-fixture-v1",
    )
    session = restore_session(config, interrupted_checkpoint, paths.repo_root)
    session.train_epoch()
    resumed = session.export_state()
    np.testing.assert_array_equal(
        resumed.input_embeddings(), final_state.input_embeddings()
    )
    np.testing.assert_array_equal(
        resumed.output_embeddings(), final_state.output_embeddings()
    )


def test_w2v1_check_and_analysis_detect_complete_result(tmp_path):
    definition = DEFINITION.get_suite("w2v1")
    spec = definition.load_run_spec(
        definition.config_root / "e01_table2_cbow.yaml",
        atomic_run_id="local-smoke",
        overrides={},
    )
    paths = _paths(tmp_path)
    run_config(
        spec.to_executor_config(),
        ExperimentContext(paths=paths),
        executor_module=definition.executor_module,
    )
    assert "complete" in check_latest(paths)
    assert "written" in analyze_latest(paths)
    assert (paths.artifacts_root / "analysis/f2/w2v1/summary.md").is_file()


def test_w2v1_tracked_interrupt_resume_and_catalog_link(tmp_path, monkeypatch):
    definition = DEFINITION.get_suite("w2v1")
    source = definition.config_root / "e01_table2_cbow.yaml"
    fixture = definition.config_root / "fixtures/corpus.txt"
    linked: list[tuple[str, str]] = []

    def materialize(config, _paths):
        config["corpus"] = {
            "path": str(fixture),
            "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        }

    @contextmanager
    def transaction(**_kwargs):
        yield object()

    class Repository:
        def __init__(self, _connection):
            pass

        def link_mlflow_run(self, slot, run_id):
            linked.append((slot, run_id))

    monkeypatch.setattr("f2.suites.w2v1.tracked.run_preflight", lambda: {})
    monkeypatch.setattr("f2.suites.w2v1.tracked._materialize_corpus", materialize)
    monkeypatch.setattr("f2.suites.w2v1.tracked.catalog_transaction", transaction)
    monkeypatch.setattr("f2.suites.w2v1.tracked.CatalogRepository", Repository)
    monkeypatch.setenv("REPRO_STAGING_ROOT", str(tmp_path / "staging"))
    monkeypatch.setenv("REPRO_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

    receipt = run_tracked_yaml(
        source,
        atomic_run_id="d50-w24m",
        executor_module=definition.executor_module,
        spec_module=definition.spec_module,
        tracking_uri=(tmp_path / "mlruns").as_uri(),
    )

    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())
    final = client.get_run(receipt.run_id)
    predecessor = final.data.tags["f2.predecessor_run_id"]
    interrupted = client.get_run(predecessor)
    assert interrupted.info.status == "KILLED"
    assert interrupted.data.tags["result.durable_complete"] == "false"
    assert final.info.status == "FINISHED"
    assert final.data.tags["result.durable_complete"] == "true"
    assert linked == [
        (
            "w2v1-reconstruction-r2-d50-w24m-s1",
            receipt.run_id,
        )
    ]
