"""End-to-end integration pipeline verification for F2 (W2V1 / W2V2).

Marked with @pytest.mark.e2e so it is skipped during standard 'just check'
and executed explicitly via 'just e2e'.

Runs a micro-scale end-to-end lifecycle:
1. Mini-corpus & benchmark creation in isolated workspace
2. Micro training execution (w2v Rust engine, 1 epoch, dim 8, 0.05s)
3. Coordinator skip / duplicate run decision check
4. MLflow publication verification (status=FINISHED, artifacts validation)
5. Analysis execution (analyze_table1_sources / analyze_table2_sources)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mlflow import MlflowClient

from f2.execution.coordinator import ExecutionDecisionKind, F2ExecutionCoordinator
from f2.execution.definitions import DEFINITION
from f2.execution.identity import ConditionIdentity
from f2.storage.analysis_cache import download_results, read_records
from f2.suites.w2v2.analysis import analyze_table1_sources
from f2.suites.w2v2.executor import W2V2Executor
from repro_core.context import RuntimePaths


@pytest.fixture
def e2e_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    suite_def = DEFINITION.get_suite("w2v2")
    repo_root = suite_def.config_root.parents[6]
    workspace_root = tmp_path / "workspace"
    paths = RuntimePaths(repo_root=repo_root, workspace_root=workspace_root)
    monkeypatch.setenv("REPRO_WORKSPACE_ROOT", str(workspace_root))
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

    mlruns = tmp_path / "mlruns"
    tracking_uri = mlruns.as_uri()
    client = MlflowClient(tracking_uri=tracking_uri)

    corpus_dir = paths.inputs("f2", "corpus")
    benchmarks_dir = paths.inputs("f2", "benchmarks")

    source = corpus_dir / "test_corpus.txt"
    sentence = (
        b"king queen man woman paris france berlin germany new_york city san_francisco "
        * 10
        + b"\n"
    )
    source.write_bytes(sentence * 5)

    word_questions = benchmarks_dir / "questions-words.txt"
    word_questions.write_bytes(
        b": capital-common-countries\nparis france berlin germany\n"
        b": family\nman woman king queen\n"
    )

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


def _make_micro_w2v2_config(
    env: dict[str, object],
    *,
    slot_id: str = "w2v2-e2e-slot-1",
    seed: int = 1,
) -> dict:
    source = env["source"]
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
                    "epochs": 1,
                    "thread_count": 1,
                    "negative_table_size": 100,
                    "objective_kind": "negative_sampling",
                    "subsampling_threshold": 0.0,
                },
            },
        )
        .with_seed(seed)
    )
    config = spec.to_executor_config()
    config["identity"]["planned_run_slot_id"] = slot_id
    config["tracking"]["experiment"] = "f2.w2v2"
    return config


@pytest.mark.e2e
def test_f2_w2v2_full_pipeline_e2e(e2e_env: dict[str, object]) -> None:
    paths: RuntimePaths = e2e_env["paths"]
    tracking_uri: str = str(e2e_env["tracking_uri"])
    client: MlflowClient = e2e_env["client"]
    executor = W2V2Executor()

    config = _make_micro_w2v2_config(e2e_env, slot_id="micro-e2e-run-01", seed=1)

    # 1. First execution: must run training and publish to MLflow
    receipt = executor.execute_spec(
        config,
        tracking_uri=tracking_uri,
        paths=paths,
        override=False,
    )
    assert not receipt.skipped
    assert receipt.run_id != ""

    run = client.get_run(receipt.run_id)
    assert run.info.status == "FINISHED"
    assert (receipt.staging_root / "result.json").is_file()
    assert (receipt.staging_root / "lookup").is_dir()
    assert (receipt.staging_root / "metrics/observations.csv").is_file()

    # 2. Second execution without override: Coordinator must SKIP
    coordinator = F2ExecutionCoordinator(tracking_uri=tracking_uri, paths=paths)
    condition = ConditionIdentity.from_config(config, suite_id="w2v2")
    decision = coordinator.decide(
        condition,
        experiment_name="f2.w2v2",
        override=False,
        atomic_run_id=str(config.get("atomic_run_id", "")),
    )
    assert decision.kind == ExecutionDecisionKind.SKIP
    assert decision.existing_run_id == receipt.run_id

    # 3. Analysis download: sync artifacts to analysis cache
    cache_root = download_results(
        tracking_uri,
        "w2v2",
        paths,
        need_evaluation_benchmarks=True,
    )
    records = read_records(cache_root)
    assert len(records) >= 1
    assert any(
        r["run_id"] == receipt.run_id and r["status"] == "FINISHED" for r in records
    )

    # 4. Table analysis generation
    outputs = analyze_table1_sources(
        cache_root,
        questions_path=e2e_env["word_questions"],
        corpus_source="wmt",
        paths=paths,
    )
    assert len(outputs) > 0
    table_dir = outputs[0]
    assert (table_dir / "summary.md").is_file()
    assert (table_dir / "summary.json").is_file()
    assert (table_dir / "runs.csv").is_file()

    summary_content = (table_dir / "summary.md").read_text(encoding="utf-8")
    assert "W2V2 Table 1" in summary_content
