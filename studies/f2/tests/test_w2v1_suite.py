from __future__ import annotations

import numpy as np

from f2.definition import DEFINITION
from f2.suites.w2v.artifacts import load_checkpoint, save_checkpoint
from f2.suites.w2v1.executor import create_session, restore_session
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
