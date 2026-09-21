from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from f2.definition import DEFINITION
from f2.suites.w2v.artifacts import load_checkpoint
from f2.suites.w2v.tracked import run_tracked_yaml
from f2.suites.w2v1.analysis import observed_training_seconds
from repro_core.cli import app
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


def _fixture_overrides() -> dict[str, object]:
    return {
        "corpus": {"path": str(Path(__file__).parent / "fixtures/w2v1-corpus.txt")},
        "vocabulary": {
            "initial_capacity": 16,
            "hash_capacity": 128,
            "min_count": 1,
            "max_lexical_words": 30_000,
        },
        "training": {
            "embedding_dimension": 8,
            "window_radius": 2,
            "epochs": 2,
            "thread_count": 1,
            "learning_rate_update_interval": 10,
            "observation_interval": 2,
        },
    }


def test_observed_training_time_sums_last_observation_per_epoch(tmp_path) -> None:
    observations = tmp_path / "observations.csv"
    observations.write_text(
        "epoch,elapsed_seconds\n1,1.5\n1,2.0\n2,3.25\n2,3.0\n",
        encoding="utf-8",
    )

    assert observed_training_seconds(observations) == 5.25


def test_w2v1_executor_completes_declared_schedule(tmp_path):
    definition = DEFINITION.get_suite("w2v1")
    assert definition.executor_module == "f2.suites.w2v1.executor"
    source = definition.config_root / "e01_table2_cbow.yaml"
    spec = definition.load_run_spec(
        source, atomic_run_id="wmt--d50-w24m", overrides=_fixture_overrides()
    ).with_seed(1)
    paths = _paths(tmp_path)

    result = run_config(
        spec.to_executor_config(),
        ExperimentContext(paths=paths),
        executor_module=definition.executor_module,
    )
    assert load_checkpoint(result.checkpoint).completed_epochs == 2
    assert result.lookup.is_dir()
    assert result.metrics.stat().st_size > 0


def test_canonical_seeds_have_distinct_staging_identities() -> None:
    definition = DEFINITION.get_suite("w2v1")
    spec = definition.load_run_spec(
        definition.config_root / "e01_table2_cbow.yaml",
        atomic_run_id="wmt--d50-w24m",
        overrides={},
    )
    assert (
        spec.with_seed(1).identity["planned_run_slot_id"]
        == "w2v1-reconstruction-r2-d50-w24m-s1"
    )
    assert (
        spec.with_seed(7).identity["planned_run_slot_id"]
        == "w2v1-reconstruction-r2-d50-w24m-s7"
    )


@pytest.mark.parametrize(
    "atomic_run_id",
    (
        "wmt--d50-w24m",
        "lm1b--d50-w24m",
        "umbc--d50-w24m",
    ),
)
def test_canonical_table2_training_conditions(atomic_run_id: str) -> None:
    definition = DEFINITION.get_suite("w2v1")
    config = definition.load_run_spec(
        definition.config_root / "e01_table2_cbow.yaml",
        atomic_run_id=atomic_run_id,
        overrides={},
    ).to_executor_config()
    assert config["training"]["initial_learning_rate"] == 0.025
    assert config["training"]["thread_count"] == 20
    assert config["training"]["window_radius"] == 4
    assert config["training"]["context_policy"] == "fixed"
    assert config["vocabulary"]["min_count"] == 1
    assert config["vocabulary"]["max_lexical_words"] == 30_000
    assert "resource_version" not in config["identity"]
    assert "corpus_manifest_digest" not in config["identity"]
    assert "evaluation" not in config


def test_evaluation_resources_do_not_change_training_identity() -> None:
    definition = DEFINITION.get_suite("w2v1")
    source = definition.config_root / "e01_table2_cbow.yaml"
    first = definition.load_run_spec(
        source,
        atomic_run_id="wmt--d50-w24m",
        overrides={"evaluation": {"questions_path": "first.txt"}},
    ).to_executor_config()
    second = definition.load_run_spec(
        source,
        atomic_run_id="wmt--d50-w24m",
        overrides={"evaluation": {"questions_path": "second.txt"}},
    ).to_executor_config()
    assert first["identity"]["config_digest"] == second["identity"]["config_digest"]


def test_training_completes_without_evaluation_dataset(tmp_path) -> None:
    definition = DEFINITION.get_suite("w2v1")
    spec = definition.load_run_spec(
        definition.config_root / "e01_table2_cbow.yaml",
        atomic_run_id="wmt--d50-w24m",
        overrides={
            **_fixture_overrides(),
            "evaluation": {"questions_path": str(tmp_path / "missing.txt")},
        },
    ).with_seed(1)
    result = run_config(
        spec.to_executor_config(),
        ExperimentContext(paths=_paths(tmp_path)),
        executor_module=definition.executor_module,
    )
    report = json.loads(result.report.read_text())
    assert report["complete"] is True
    assert "evaluation" not in report
    assert load_checkpoint(result.checkpoint).completed_epochs == 2


def test_canonical_cli_requires_explicit_large_run_approval() -> None:
    result = CliRunner().invoke(
        app,
        [
            "run",
            "f2",
            "w2v1",
            "-e",
            "01",
            "-a",
            "wmt--d50-w24m",
            "--seed",
            "1",
            "--tracking-uri",
            "http://127.0.0.1:1",
        ],
    )
    assert result.exit_code == 2
    assert "requires --approve-large-run" in result.output


@pytest.mark.parametrize(
    ("suite", "config_name", "atomic_run_id", "expected_slot"),
    (
        (
            "w2v1",
            "e01_table2_cbow.yaml",
            "wmt--d50-w24m",
            "w2v1-reconstruction-r2-d50-w24m-s1",
        ),
        (
            "w2v2",
            "e02_table3_phrase_skipgram.yaml",
            "wmt--neg5-subsampling",
            "w2v2-reconstruction-r1-neg5-subsampling-s1",
        ),
    ),
)
def test_w2v_tracked_run_publishes_one_complete_mlflow_run(
    tmp_path, monkeypatch, suite, config_name, atomic_run_id, expected_slot
):
    definition = DEFINITION.get_suite(suite)
    source = definition.config_root / config_name
    fixture = Path(__file__).parent / f"fixtures/{suite}-corpus.txt"

    def materialize(config, _paths, **_kwargs):
        config["corpus"] = {
            "path": str(fixture),
            "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        }
        if suite == "w2v2":
            config["vocabulary"] = {
                "initial_capacity": 16,
                "hash_capacity": 128,
                "min_count": 1,
            }
            config["training"] = {
                **config["training"],
                "embedding_dimension": 8,
                "negative_table_size": 100,
            }

    monkeypatch.setattr("f2.suites.w2v.tracked._materialize_corpus", materialize)
    monkeypatch.setenv("REPRO_STAGING_ROOT", str(tmp_path / "staging"))
    monkeypatch.setenv("REPRO_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

    receipt = run_tracked_yaml(
        source,
        atomic_run_id=atomic_run_id,
        executor_module=definition.executor_module,
        spec_module=definition.spec_module,
        tracking_uri=(tmp_path / "mlruns").as_uri(),
        overrides=_fixture_overrides()
        if suite == "w2v1"
        else {
            "vocabulary": {
                "initial_capacity": 16,
                "hash_capacity": 128,
                "min_count": 1,
            },
            "training": {
                "embedding_dimension": 8,
                "epochs": 2,
                "thread_count": 1,
                "negative_table_size": 100,
            },
            "phrase_detection": {"passes": 1, "threshold": 0.0, "min_count": 1},
        },
    )

    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())
    runs = client.search_runs([client.get_run(receipt.run_id).info.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FINISHED"
    assert run.data.tags["mlflow.runName"] == expected_slot
    assert run.data.tags["result.durable_complete"] == "true"
    assert run.data.tags["suite.name"] == suite
    assert "f2.resource_manifest_digest" not in run.data.tags
    assert "f2.resource_version_id" not in run.data.tags
    assert run.data.tags["f2.planned_run_slot_id"] == expected_slot
    assert "f2.attempt" not in run.data.tags
    assert "f2.predecessor_run_id" not in run.data.tags
