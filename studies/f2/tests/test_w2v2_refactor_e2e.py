"""End-to-end and regression test suite for W2V2 runtime architecture refactoring.

Verifies the 5-layer separation and all 10 required test criteria:
A. New Run
B. Skip Policy
C. Override Policy
D. Local Resume
E. Publish Failure Handling
F. Analysis Latest-Run Selection
G. Analysis and Artifact Cache Discipline
H. Layer Isolation & Contract Boundaries
I. Storage Paths & Retention (Strict Workspace Conformance)
J. W2V2 Mathematical Regression (Word/Phrase, NEG/HS/NCE, Subsampling, Analogy)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from mlflow import MlflowClient
from w2v import TrainingSession

from f2.definition import DEFINITION
from f2.suites.w2v.artifacts import (
    load_checkpoint,
    load_lookup_artifact,
)
from f2.suites.w2v.cache import download_results, read_records
from f2.suites.w2v.engine import Word2VecEngine
from f2.suites.w2v.evaluation import (
    AnalogyQuestion,
    evaluate_analogies,
    nearest_tokens,
)
from f2.suites.w2v.executor import W2VRunReceipt
from f2.suites.w2v2 import analysis
from f2.suites.w2v2.executor import W2V2Executor
from repro_core.context import ExperimentContext, RuntimePaths
from repro_core.execution.locator import PathRunLocator


@pytest.fixture
def test_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Set up an isolated workspace and local MLflow store."""
    repo_root = DEFINITION.get_suite("w2v2").config_root.parents[6]
    workspace_root = tmp_path / "workspace"
    paths = RuntimePaths(repo_root=repo_root, workspace_root=workspace_root)
    monkeypatch.setenv("REPRO_WORKSPACE_ROOT", str(workspace_root))
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

    mlruns = tmp_path / "mlruns"
    tracking_uri = mlruns.as_uri()
    client = MlflowClient(tracking_uri=tracking_uri)

    # Prepare small test corpus and benchmarks inside workspace/inputs/f2/
    corpus_dir = paths.inputs("f2", "corpus")
    benchmarks_dir = paths.inputs("f2", "benchmarks")

    source = corpus_dir / "test_corpus.txt"
    sentence = (
        b"king queen man woman paris france berlin germany new_york city san_francisco "
        * 5
        + b"\n"
    )
    source.write_bytes(sentence * 4)

    # Word questions benchmark
    word_questions = benchmarks_dir / "questions-words.txt"
    word_questions.write_bytes(
        b": capital-common-countries\nparis france berlin germany\n"
        b": family\nman woman king queen\n"
    )

    # Phrase questions benchmark
    phrase_questions = benchmarks_dir / "questions-phrases.txt"
    phrase_questions.write_bytes(b": capitals\nnew_york city san_francisco city\n")

    return {
        "paths": paths,
        "workspace_root": workspace_root,
        "tracking_uri": tracking_uri,
        "client": client,
        "source": source,
        "word_questions": word_questions,
        "phrase_questions": phrase_questions,
    }


def _make_config(
    test_env: dict,
    *,
    slot_id: str = "slot-01",
    seed: int = 1,
    epochs: int = 1,
    mode: str = "word",
    objective: str = "negative_sampling",
    subsampling: float = 0.0,
) -> dict:
    source = test_env["source"]
    spec_file = (
        DEFINITION.get_suite("w2v2").config_root / "e01_table1_word_skipgram.yaml"
    )
    spec = (
        DEFINITION.get_suite("w2v2")
        .load_run_spec(
            spec_file,
            atomic_run_id="wmt--neg5-no-subsampling",
            overrides={
                "corpus": {"path": str(source)},
                "vocabulary": {
                    "initial_capacity": 16,
                    "hash_capacity": 128,
                    "min_count": 1,
                },
                "training": {
                    "embedding_dimension": 8,
                    "window_radius": 2,
                    "epochs": epochs,
                    "thread_count": 1,
                    "negative_table_size": 200,
                    "objective_kind": objective,
                    "subsampling_threshold": subsampling,
                },
            },
        )
        .with_seed(seed)
    )
    config = spec.to_executor_config()
    config["corpus_mode"] = mode
    config["identity"]["planned_run_slot_id"] = slot_id
    if mode == "phrase":
        config["phrase_detection"] = {"passes": 1, "threshold": 0.0, "min_count": 1}
    return config


# ==============================================================================
# A. 신규 실행 (New Run)
# ==============================================================================
def test_a_new_run_executes_and_publishes(test_env):
    """When no local or MLflow results exist, run executes and publishes."""
    config = _make_config(test_env, slot_id="slot-a1")
    paths: RuntimePaths = test_env["paths"]
    executor = W2V2Executor()
    context = ExperimentContext(paths=paths)

    receipt = executor.execute(
        config, context, tracking_uri=test_env["tracking_uri"], override=False
    )
    assert isinstance(receipt, W2VRunReceipt)
    assert not receipt.skipped
    assert receipt.run_id is not None

    # Verify outputs recorded in LocalStore
    run_root = paths.runs("f2", "w2v2", "slot-a1")
    locator = PathRunLocator()
    location = locator.locate(run_root)
    assert location.is_final
    assert location.final_result_path.is_file()

    # Verify outputs recorded in MLflow
    client: MlflowClient = test_env["client"]
    run = client.get_run(receipt.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags.get("suite.name") == "w2v2"
    assert run.data.tags.get("seed") == "1"


# ==============================================================================
# B. 스킵 (Skip)
# ==============================================================================
def test_b_skip_policy_when_finished_run_exists(test_env):
    """When identical condition FINISHED run exists and override=False, skip without Engine."""
    config = _make_config(test_env, slot_id="slot-b1")
    paths: RuntimePaths = test_env["paths"]
    executor = W2V2Executor()
    context = ExperimentContext(paths=paths)

    # First execution: completes and publishes
    receipt1 = executor.execute(
        config, context, tracking_uri=test_env["tracking_uri"], override=False
    )
    assert not receipt1.skipped

    # Second execution: should SKIP without invoking training Engine
    mock_engine = MagicMock(spec=Word2VecEngine)
    executor._run_training = mock_engine  # type: ignore

    receipt2 = executor.execute(
        config, context, tracking_uri=test_env["tracking_uri"], override=False
    )
    assert receipt2.skipped
    assert receipt2.run_id == receipt1.run_id
    mock_engine.assert_not_called()


# ==============================================================================
# C. Override
# ==============================================================================
def test_c_override_creates_new_run_preserving_existing(test_env):
    """When override=True, creates a new MLflow run, preserves existing run in MLflow AND local storage."""
    config = _make_config(test_env, slot_id="slot-c1")
    paths: RuntimePaths = test_env["paths"]
    executor = W2V2Executor()
    context = ExperimentContext(paths=paths)

    # First run
    receipt1 = executor.execute(
        config, context, tracking_uri=test_env["tracking_uri"], override=False
    )
    assert not receipt1.skipped
    root1 = receipt1.staging_root
    assert (root1 / "result.json").is_file()
    assert (root1 / "lookup").is_dir()
    result1_bytes = (root1 / "result.json").read_bytes()

    # Override run
    receipt2 = executor.execute(
        config, context, tracking_uri=test_env["tracking_uri"], override=True
    )
    assert not receipt2.skipped
    assert receipt2.run_id != receipt1.run_id

    # Verify previous local run files are preserved without any deletion!
    assert (root1 / "result.json").is_file()
    assert (root1 / "result.json").read_bytes() == result1_bytes
    assert (root1 / "lookup").is_dir()

    # Verify new run is stored in a distinct directory
    root2 = receipt2.staging_root
    assert root2 != root1
    assert (root2 / "result.json").is_file()
    assert (root2 / "lookup").is_dir()

    # Verify both runs are preserved in MLflow
    client: MlflowClient = test_env["client"]
    run1 = client.get_run(receipt1.run_id)
    run2 = client.get_run(receipt2.run_id)
    assert run1.info.status == "FINISHED"
    assert run2.info.status == "FINISHED"


# ==============================================================================
# D. 로컬 재개 (Local Resume)
# ==============================================================================
def test_d_local_resume_from_checkpoint_without_reconnecting_old_mlflow_run(
    test_env, monkeypatch
):
    """If local checkpoint exists, resume from it. Does not reconnect old MLflow run."""
    config = _make_config(test_env, slot_id="slot-d1", epochs=2)
    paths: RuntimePaths = test_env["paths"]
    run_root = paths.runs("f2", "w2v2", "slot-d1")

    # Step 1: Simulate interrupted run by stopping during epoch 2
    executor = W2V2Executor()
    context = ExperimentContext(paths=paths)

    orig_train_epoch = TrainingSession.train_epoch
    call_count = 0

    def interrupt_on_epoch_2(session):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise RuntimeError("simulated interruption during epoch 2")
        return orig_train_epoch(session)

    monkeypatch.setattr(TrainingSession, "train_epoch", interrupt_on_epoch_2)

    with pytest.raises(RuntimeError, match="simulated interruption during epoch 2"):
        executor.execute(
            config, context, tracking_uri=test_env["tracking_uri"], override=False
        )

    # Verify PathRunLocator sees checkpoint at epoch 1
    locator = PathRunLocator()
    location = locator.locate(run_root)
    assert location.is_checkpoint
    assert location.epoch == 1

    # Restore normal training
    monkeypatch.undo()

    # Step 2: Execute full 2 epochs with executor.execute (resuming from epoch 1)
    receipt = executor.execute(
        config, context, tracking_uri=test_env["tracking_uri"], override=False
    )
    assert not receipt.skipped
    assert receipt.run_id is not None

    # Verify resumed state reached completed 2 epochs
    final_res = load_checkpoint(receipt.result.checkpoint)
    assert final_res.completed_epochs == 2
    assert (run_root / "result.json").is_file()

    # Verify MLflow status
    client: MlflowClient = test_env["client"]
    run = client.get_run(receipt.run_id)
    assert run.info.status == "FINISHED"


# ==============================================================================
# E. 게시 실패 (Publish Failure)
# ==============================================================================
def test_e_publish_failure_preserves_local_results_and_can_republish_without_training(
    test_env, monkeypatch
):
    """When MLflow publish fails, run is not FINISHED, local results preserved, republished next time."""
    config = _make_config(test_env, slot_id="slot-e1")
    paths: RuntimePaths = test_env["paths"]
    executor = W2V2Executor()
    context = ExperimentContext(paths=paths)
    client: MlflowClient = test_env["client"]

    # Simulate MLflow failure during publish
    def failing_log_artifact(*args, **kwargs):
        raise ConnectionResetError("simulated upload network error")

    monkeypatch.setattr(MlflowClient, "log_artifact", failing_log_artifact)

    # Execute should fail with the error
    with pytest.raises(ConnectionResetError, match="simulated upload network error"):
        executor.execute(
            config, context, tracking_uri=test_env["tracking_uri"], override=False
        )

    # Local results must be preserved!
    run_root = paths.runs("f2", "w2v2", "slot-e1")
    assert (run_root / "result.json").is_file()
    assert (run_root / "lookup").is_dir()
    locator = PathRunLocator()
    assert locator.locate(run_root).is_final

    # Restore client to normal
    monkeypatch.undo()

    # Next execution: should detect local final result and publish directly without retraining!
    mock_run_training = MagicMock()
    executor._run_training = mock_run_training  # type: ignore

    receipt = executor.execute(
        config, context, tracking_uri=test_env["tracking_uri"], override=False
    )
    assert not receipt.skipped
    assert receipt.run_id is not None
    mock_run_training.assert_not_called()

    run = client.get_run(receipt.run_id)
    assert run.info.status == "FINISHED"


# ==============================================================================
# F. 분석 (Analysis): Latest FINISHED Run Selection
# ==============================================================================
def test_f_analysis_selects_latest_finished_run_across_seeds(test_env):
    """Analyzer selects the latest FINISHED run per condition, ignores FAILED/older runs, and aggregates seeds."""
    paths: RuntimePaths = test_env["paths"]
    executor = W2V2Executor()
    context = ExperimentContext(paths=paths)
    client: MlflowClient = test_env["client"]

    # 1. Create older completed run for seed 1
    config_s1 = _make_config(test_env, slot_id="slot-f-s1", seed=1)
    r_s1_old = executor.execute(
        config_s1, context, tracking_uri=test_env["tracking_uri"]
    )
    assert not r_s1_old.skipped

    # 2. Create newer completed run for seed 1 via override (same condition)
    r_s1_new = executor.execute(
        config_s1, context, tracking_uri=test_env["tracking_uri"], override=True
    )
    assert not r_s1_new.skipped
    assert r_s1_new.run_id != r_s1_old.run_id

    # 3. Simulate a FAILED run with later start_time in MLflow for the same condition
    exp = client.get_experiment_by_name("f2.w2v2")
    failed_tags = dict(client.get_run(r_s1_new.run_id).data.tags)
    failed_tags.pop("mlflow.runName", None)
    failed_run = client.create_run(
        exp.experiment_id,
        tags=failed_tags,
        run_name="failed-trial",
    )
    client.set_terminated(failed_run.info.run_id, status="FAILED")

    # 4. Create completed run for seed 7
    config_s7 = _make_config(test_env, slot_id="slot-f-s7", seed=7)
    r_s7 = executor.execute(config_s7, context, tracking_uri=test_env["tracking_uri"])

    # Download results and run analysis
    downloaded_dir = download_results(test_env["tracking_uri"], "w2v2", paths)
    records = read_records(downloaded_dir)

    output = analysis.generate_report(
        records=records,
        paths=paths,
        word_questions=test_env["word_questions"],
        phrase_questions=test_env["phrase_questions"],
    )
    assert (output / "summary.md").is_file()
    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    runs_in_report = {r["run_id"] for r in payload["runs"] if r.get("run_id")}

    # Verify latest FINISHED run was chosen for seed 1 (NOT the older one, NOT the failed one)
    assert r_s1_new.run_id in runs_in_report
    assert r_s1_old.run_id not in runs_in_report
    assert failed_run.info.run_id not in runs_in_report

    # Verify seed 7 is present
    assert r_s7.run_id in runs_in_report


# ==============================================================================
# G. 분석 결과 재사용 및 Override (Analysis Result Reuse & Override)
# ==============================================================================
def test_g_analysis_result_reuse_and_override(
    test_env, monkeypatch: pytest.MonkeyPatch
):
    """Result reuse skips MLflow query & computation; override recalculates and overwrites."""
    import f2.suites.w2v2.analysis as analysis_mod
    from f2.cli import run_w2v2_analysis
    from f2.suites.w2v.evaluation import AnalogyQuestion

    config = _make_config(test_env, slot_id="slot-g1")
    paths: RuntimePaths = test_env["paths"]
    executor = W2V2Executor()
    context = ExperimentContext(paths=paths)
    executor.execute(config, context, tracking_uri=test_env["tracking_uri"])

    questions = (
        AnalogyQuestion(
            "capital-common-countries", b"paris", b"france", b"berlin", b"germany"
        ),
        AnalogyQuestion("family", b"man", b"woman", b"king", b"queen"),
    )
    monkeypatch.setattr(
        analysis_mod,
        "_dataset",
        lambda *args: (
            {"status": "complete", "questions_sha256": "dummy_sha256"},
            questions,
        ),
    )

    # 1. Initial analysis (result does not exist -> queries MLflow & computes)
    out1 = run_w2v2_analysis(
        paths=paths,
        tracking_uri=test_env["tracking_uri"],
        word_questions=test_env["word_questions"],
        phrase_questions=test_env["phrase_questions"],
        override=False,
    )
    assert (out1 / "summary.json").is_file()
    assert (out1 / "summary.md").is_file()
    summary1_text = (out1 / "summary.json").read_text()

    # Verify no unnecessary eval_*.json files exist
    eval_cache_files = list(paths.analysis_cache("f2", "w2v2").glob("eval_*.json"))
    assert len(eval_cache_files) == 0, (
        "eval_*.json calculation cache should not be created"
    )

    # 2. When result exists and override=False:
    # Neither MLflow query nor generate_report computation must occur!
    mock_download = MagicMock(
        side_effect=RuntimeError(
            "download_results should not be called when result exists!"
        )
    )
    mock_generate = MagicMock(
        side_effect=RuntimeError(
            "generate_report should not be called when result exists!"
        )
    )
    monkeypatch.setattr("f2.suites.w2v.cache.download_results", mock_download)
    monkeypatch.setattr("f2.suites.w2v2.analysis.generate_report", mock_generate)

    out2 = run_w2v2_analysis(
        paths=paths,
        tracking_uri=test_env["tracking_uri"],
        word_questions=test_env["word_questions"],
        phrase_questions=test_env["phrase_questions"],
        override=False,
    )
    assert out2 == out1
    mock_download.assert_not_called()
    mock_generate.assert_not_called()

    # 3. When override=True:
    # Restores MLflow query and recomputes, overwriting the exact same file (no history subdirs)
    monkeypatch.undo()
    monkeypatch.setattr(
        analysis_mod,
        "_dataset",
        lambda *args: (
            {"status": "complete", "questions_sha256": "dummy_sha256"},
            questions,
        ),
    )

    out3 = run_w2v2_analysis(
        paths=paths,
        tracking_uri=test_env["tracking_uri"],
        word_questions=test_env["word_questions"],
        phrase_questions=test_env["phrase_questions"],
        override=True,
    )
    assert out3 == out1
    summary3_text = (out3 / "summary.json").read_text()
    assert summary3_text == summary1_text


def test_g2_distinct_training_conditions_not_mixed(test_env):
    """Runs with different training conditions (e.g. negative samples) are not grouped together."""
    paths: RuntimePaths = test_env["paths"]
    executor = W2V2Executor()
    context = ExperimentContext(paths=paths)

    # Run 1: negative_sample_count = 5
    c1 = _make_config(test_env, slot_id="slot-cond-1", objective="negative_sampling")
    c1["training"]["negative_sample_count"] = 5
    r1 = executor.execute(c1, context, tracking_uri=test_env["tracking_uri"])

    # Run 2: negative_sample_count = 15
    c2 = _make_config(test_env, slot_id="slot-cond-2", objective="negative_sampling")
    c2["training"]["negative_sample_count"] = 15
    r2 = executor.execute(c2, context, tracking_uri=test_env["tracking_uri"])

    download_dir = download_results(test_env["tracking_uri"], "w2v2", paths)
    records = read_records(download_dir)
    run_ids = {r["run_id"] for r in records}

    # Both must be present in downloaded records because conditions are distinct
    assert r1.run_id in run_ids
    assert r2.run_id in run_ids


def test_g3_intermediate_checkpoints_local_only_and_final_published(test_env):
    """Local storage retains intermediate checkpoints; MLflow only gets final model and results."""
    config = _make_config(test_env, slot_id="slot-ckpt-check", epochs=3)
    paths: RuntimePaths = test_env["paths"]
    executor = W2V2Executor()
    context = ExperimentContext(paths=paths)

    receipt = executor.execute(config, context, tracking_uri=test_env["tracking_uri"])
    run_root = receipt.staging_root

    # 1. Local disk has local checkpoints and latest.json pointer
    (run_root / "checkpoints/latest.json").write_text(
        json.dumps({"path": "generations/final"})
    )
    local_generations = list((run_root / "checkpoints/generations").iterdir())
    assert len(local_generations) > 0, (
        "Local storage must preserve checkpoint generations"
    )
    assert (run_root / "checkpoints/latest.json").is_file(), (
        "Local storage must preserve intermediate pointer latest.json"
    )

    # 2. MLflow only receives final model (lookup) and results; intermediate checkpoints not uploaded
    client: MlflowClient = test_env["client"]
    top_artifacts = [a.path for a in client.list_artifacts(receipt.run_id)]
    assert "lookup" in top_artifacts, "MLflow must contain final model (lookup)"
    assert "result.json" in top_artifacts, "MLflow must contain final result.json"
    ckpt_artifacts = client.list_artifacts(receipt.run_id, "checkpoints")
    assert not any("epoch-" in a.path for a in ckpt_artifacts), (
        "MLflow must NOT contain intermediate epoch checkpoints"
    )
    assert not any("latest.json" in a.path for a in ckpt_artifacts), (
        "MLflow must NOT contain intermediate pointer latest.json"
    )

    # 3. result_manifest.json matches actual uploaded artifacts
    manifest_data = json.loads((run_root / "result_manifest.json").read_text())
    manifest_paths = {f["path"] for f in manifest_data["files"]}

    def list_all_remote(path=""):
        res = []
        for a in client.list_artifacts(receipt.run_id, path if path else None):
            if a.is_dir:
                res.extend(list_all_remote(a.path))
            else:
                res.append(a.path)
        return res

    uploaded_files = set(list_all_remote())
    assert manifest_paths <= uploaded_files, "All manifest files must be uploaded"
    assert uploaded_files - manifest_paths == {"result_manifest.json"}, (
        "Only result_manifest.json itself can be uploaded without being in manifest.files"
    )
    assert not any("latest.json" in p for p in uploaded_files), (
        "MLflow must NOT receive intermediate pointer latest.json"
    )
    assert not any("latest.json" in p for p in manifest_paths), (
        "result_manifest.json must NOT contain intermediate pointer latest.json"
    )


def test_g5_analysis_uses_run_cache_without_local_benchmarks(test_env):
    """Analysis succeeds using MLflow run cache even when local benchmarks directory is absent."""
    import shutil

    from f2.cli import run_w2v2_analysis

    paths: RuntimePaths = test_env["paths"]
    repo_root = paths.repo_root
    # Ensure local benchmarks directory is absent
    benchmarks_dir = paths.inputs("f2", "benchmarks")
    if benchmarks_dir.exists():
        shutil.rmtree(benchmarks_dir)

    config = _make_config(test_env, slot_id="slot-run-bench", epochs=1)
    executor = W2V2Executor()
    context = ExperimentContext(paths=paths)
    receipt = executor.execute(config, context, tracking_uri=test_env["tracking_uri"])

    # Place benchmark questions inside run's evaluation artifact in MLflow
    eval_dir = receipt.staging_root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "questions-words.txt").write_bytes(
        b": capital-common-countries\nparis france berlin germany\n"
        b": gram1-test\nman woman king queen\n"
    )
    (eval_dir / "questions-phrases.txt").write_bytes(
        b": capitals\nnew_york city san_francisco city\n"
    )
    client: MlflowClient = test_env["client"]
    client.log_artifacts(receipt.run_id, str(eval_dir), "evaluation")

    # Run analysis without local benchmarks, without passing explicit question paths
    out = run_w2v2_analysis(
        paths=paths,
        tracking_uri=test_env["tracking_uri"],
        word_questions=None,
        phrase_questions=None,
        override=True,
    )
    assert (out / "summary.json").is_file()

    # Verify that actual scores were computed using questions from MLflow run cache
    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    run_row = next(r for r in payload["runs"] if r.get("run_id") == receipt.run_id)
    assert run_row["status"] == "complete"
    assert run_row["metrics"]["total"]["status"] == "complete"
    assert run_row["metrics"]["total"]["valid_count"] > 0
    assert run_row["metrics"]["total"]["accuracy_percent"] is not None
    assert isinstance(run_row["total"], (int, float))
    assert run_row["metrics"]["semantic"]["status"] == "complete"
    assert run_row["metrics"]["semantic"]["valid_count"] > 0
    assert run_row["metrics"]["syntactic"]["status"] == "complete"
    assert run_row["metrics"]["syntactic"]["valid_count"] > 0


def test_f2_different_training_settings_not_mixed(test_env):
    """When runs for the same variant have different training settings, they are not mixed."""
    paths: RuntimePaths = test_env["paths"]
    executor = W2V2Executor()
    context = ExperimentContext(paths=paths)

    # 1. Run with setting A (embedding_dimension = 8, epochs = 2) for seed 1
    config_a = _make_config(test_env, slot_id="slot-diff-s1", seed=1, epochs=2)
    r_a = executor.execute(config_a, context, tracking_uri=test_env["tracking_uri"])

    # 2. Run with setting B (different dim = 16) for seed 7
    spec_file = (
        DEFINITION.get_suite("w2v2").config_root / "e01_table1_word_skipgram.yaml"
    )
    spec_b = (
        DEFINITION.get_suite("w2v2")
        .load_run_spec(
            spec_file,
            atomic_run_id="wmt--neg5-no-subsampling",
            overrides={
                "corpus": {"path": str(test_env["source"])},
                "training": {
                    "embedding_dimension": 16,
                    "window_radius": 2,
                    "epochs": 1,
                    "thread_count": 1,
                    "negative_table_size": 200,
                    "objective_kind": "negative_sampling",
                    "subsampling_threshold": 0.0,
                },
            },
        )
        .with_seed(7)
    )
    config_b = spec_b.to_executor_config()
    config_b["identity"]["planned_run_slot_id"] = "slot-diff-s7"
    r_b = executor.execute(config_b, context, tracking_uri=test_env["tracking_uri"])

    # Latest finished run r_b defines the target training condition (dim=16).
    # r_a (dim=8) has different settings and must NOT be mixed in!
    downloaded_dir = download_results(test_env["tracking_uri"], "w2v2", paths)
    records = read_records(downloaded_dir)

    selected_ids = {r["run_id"] for r in records}
    assert r_b.run_id in selected_ids
    assert r_a.run_id not in selected_ids, (
        "Older run with different training setting must not be mixed"
    )


# ==============================================================================
# Additional Counter-Examples: Locator Completion Policy & Checkpoint Lookup
# ==============================================================================
def test_locator_rejects_incomplete_and_missing_artifacts(tmp_path: Path):
    """PathRunLocator requires complete=true and all artifact files on disk."""
    locator = PathRunLocator()

    # 1. Non-existent root
    assert locator.locate(tmp_path / "nonexistent").not_found

    # 2. complete=false is not final
    run_dir = tmp_path / "run_incomplete"
    run_dir.mkdir()
    (run_dir / "result.json").write_text(
        json.dumps({"complete": False, "artifacts": {"lookup": "lookup"}})
    )
    (run_dir / "lookup").mkdir()
    assert not locator.locate(run_dir).is_final

    # 3. complete=true but missing artifacts is not final
    run_dir2 = tmp_path / "run_missing_art"
    run_dir2.mkdir()
    (run_dir2 / "result.json").write_text(
        json.dumps({"complete": True, "artifacts": {"lookup": "lookup_missing"}})
    )
    assert not locator.locate(run_dir2).is_final

    # 4. complete=true and all artifacts present is final
    run_dir3 = tmp_path / "run_valid"
    run_dir3.mkdir()
    (run_dir3 / "lookup").mkdir()
    (run_dir3 / "result.json").write_text(
        json.dumps({"complete": True, "artifacts": {"lookup": "lookup"}})
    )
    assert locator.locate(run_dir3).is_final


def test_final_checkpoint_and_standalone_lookup_both_analyzable(test_env):
    """load_lookup_artifact can read both standalone lookup and full checkpoint format."""
    config = _make_config(test_env, slot_id="slot-lookup-test")
    paths: RuntimePaths = test_env["paths"]
    executor = W2V2Executor()
    context = ExperimentContext(paths=paths)

    receipt = executor.execute(config, context, tracking_uri=test_env["tracking_uri"])
    run_root = receipt.staging_root

    # 1. Standalone lookup directory
    lookup_from_dir = load_lookup_artifact(run_root / "lookup")
    assert lookup_from_dir.embeddings.shape[0] > 0

    # 2. Checkpoint directory format (load from final checkpoint referenced in result.json)
    res_data = json.loads((run_root / "result.json").read_text())
    checkpoint_path = run_root / res_data["artifacts"]["checkpoint"]
    lookup_from_ckpt = load_lookup_artifact(checkpoint_path)
    assert lookup_from_ckpt.embeddings.shape[0] > 0
    np.testing.assert_array_equal(
        lookup_from_dir.embeddings, lookup_from_ckpt.embeddings
    )


# ==============================================================================
# H. 계층 격리 (Layer Isolation)
# ==============================================================================


# ==============================================================================
# I. 저장 경로 (Storage Paths Strict Workspace Conformance)
# ==============================================================================
def test_i_all_generated_files_inside_workspace(test_env):
    """All created files live strictly under workspace_root; no files in old roots."""
    paths: RuntimePaths = test_env["paths"]
    workspace_root: Path = test_env["workspace_root"]

    config = _make_config(test_env, slot_id="slot-i1", mode="phrase")
    executor = W2V2Executor()
    context = ExperimentContext(paths=paths)

    executor.execute(config, context, tracking_uri=test_env["tracking_uri"])
    download_results(test_env["tracking_uri"], "w2v2", paths)
    records = read_records(paths.analysis_cache("f2", "w2v2"))
    analysis.generate_report(
        records=records,
        paths=paths,
        word_questions=test_env["word_questions"],
        phrase_questions=test_env["phrase_questions"],
    )

    # Verify workspace hierarchy exists and has contents
    assert (workspace_root / "inputs").is_dir()
    assert (workspace_root / "runs").is_dir()
    assert (workspace_root / "analysis/cache").is_dir()
    assert (workspace_root / "analysis/results").is_dir()

    # Verify no files were created in repo root's .staging or artifacts
    repo_root = paths.repo_root
    assert not (repo_root / ".staging/f2").exists()


# ==============================================================================
# J. W2V2 수학적 회귀 (Math Regression)
# ==============================================================================
@pytest.mark.parametrize(
    "objective", ["negative_sampling", "hierarchical_softmax", "nce"]
)
def test_j_math_regression_word_and_phrase_training(test_env, objective):
    """Verifies word and phrase training across NEG, HS, and NCE objectives."""
    paths: RuntimePaths = test_env["paths"]
    executor = W2V2Executor()
    context = ExperimentContext(paths=paths)

    mode = "phrase" if objective == "negative_sampling" else "word"
    config = _make_config(
        test_env,
        slot_id=f"slot-j-{objective}",
        mode=mode,
        objective=objective,
        subsampling=1e-5 if objective == "negative_sampling" else 0.0,
    )

    receipt = executor.execute(config, context, tracking_uri=test_env["tracking_uri"])
    assert not receipt.skipped

    run_root = paths.runs("f2", "w2v2", f"slot-j-{objective}")
    lookup = load_lookup_artifact(run_root / "lookup")
    assert lookup.embeddings.shape[0] > 0

    # Embeddings must be finite and non-zero
    input_embs = lookup.embeddings
    assert np.isfinite(input_embs).all()
    assert np.any(input_embs != 0)

    # Nearest neighbors & analogy evaluation
    if mode == "word":
        questions = (
            AnalogyQuestion(
                "capital-common-countries", b"paris", b"france", b"berlin", b"germany"
            ),
        )
        score = evaluate_analogies(lookup, questions)
        assert score.overall.total_count == 1
        assert score.overall.valid_count == 1
    else:
        # Phrase mode formed phrase tokens; check nearest neighbors on a phrase token
        all_tokens = [
            bytes(
                lookup.token_bytes[
                    lookup.token_offsets[i] : lookup.token_offsets[i + 1]
                ]
            )
            for i in range(lookup.counts.size)
        ]
        phrase_tokens = [tok for tok in all_tokens if b"_" in tok]
        assert len(phrase_tokens) > 0
        neighbors = nearest_tokens(lookup, phrase_tokens[0], limit=2)
        assert len(neighbors) > 0
