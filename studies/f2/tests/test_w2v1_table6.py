from __future__ import annotations

import json
from pathlib import Path

import pytest
from w2v import DownpourTrainingSession, TrainingSession

from f2.definition import DEFINITION
from f2.suites.w2v.artifacts import load_checkpoint, load_lookup_artifact
from f2.suites.w2v1.executor import create_session
from repro_core.context import ExperimentContext, RuntimePaths
from repro_core.execution.definition import RunOptions, RunSelection
from repro_core.execution.planning import Planner
from repro_core.execution.runner import run_config


def test_table6_plan_and_spec_structure() -> None:
    definition = DEFINITION.get_suite("w2v1")
    plans = Planner(definition).build(
        RunSelection(experiment_ids=("05",)), RunOptions()
    )
    assert len(plans) == 6  # 2 variants x 3 seeds (1, 7, 19)

    for plan in plans:
        spec = definition.load_run_spec(
            plan.path, atomic_run_id=plan.atomic_run_id, overrides={}
        ).with_seed(plan.seed)
        config = spec.to_executor_config()
        identity = config["identity"]

        assert identity["study"] == "table6"
        assert identity["execution_plan_id"] == "w2v1-fineweb-reconstruction-r1"
        assert config["vocabulary"]["max_lexical_words"] == 1_000_000

        training = config["training"]
        assert training["thread_count"] == 14
        assert training["embedding_dimension"] == 1000
        assert training["epochs"] == 1
        assert training["objective_kind"] == "hierarchical_softmax"

        distribution = config.get("distribution")
        assert distribution is not None
        assert distribution["mode"] == "downpour"
        assert distribution["parameter_server_shards"] == 4
        assert distribution["mini_batch_targets"] == 250
        assert distribution["adagrad_gamma"] == 0.05
        assert distribution["adagrad_epsilon"] == 1.0e-6
        assert distribution["queue_capacity"] == 128


def test_table6_distribution_digest_changes_identity() -> None:
    definition = DEFINITION.get_suite("w2v1")
    source = definition.config_root / "e05_table6.yaml"

    spec_orig = definition.load_run_spec(
        source, atomic_run_id="fineweb--cbow-d1000-w6000m", overrides={}
    ).to_executor_config()

    spec_mod_shards = definition.load_run_spec(
        source,
        atomic_run_id="fineweb--cbow-d1000-w6000m",
        overrides={"distribution": {"parameter_server_shards": 8}},
    ).to_executor_config()

    spec_mod_batch = definition.load_run_spec(
        source,
        atomic_run_id="fineweb--cbow-d1000-w6000m",
        overrides={"distribution": {"mini_batch_targets": 200}},
    ).to_executor_config()

    digest_orig = spec_orig["identity"]["config_digest"]
    digest_shards = spec_mod_shards["identity"]["config_digest"]
    digest_batch = spec_mod_batch["identity"]["config_digest"]

    assert digest_orig != digest_shards
    assert digest_orig != digest_batch
    assert digest_shards != digest_batch


def test_create_session_dispatch(tmp_path: Path) -> None:
    definition = DEFINITION.get_suite("w2v1")
    fixture_corpus = Path(__file__).parent / "fixtures/w2v1-corpus.txt"

    base_overrides = {
        "corpus": {"path": str(fixture_corpus)},
        "vocabulary": {
            "initial_capacity": 16,
            "hash_capacity": 128,
            "min_count": 1,
            "max_lexical_words": 100,
        },
        "training": {
            "embedding_dimension": 8,
            "epochs": 1,
            "thread_count": 2,
            "learning_rate_update_interval": 5,
            "observation_interval": 1,
        },
    }

    # Downpour config produces DownpourTrainingSession
    spec_downpour = definition.load_run_spec(
        definition.config_root / "e05_table6.yaml",
        atomic_run_id="fineweb--cbow-d1000-w6000m",
        overrides={
            **base_overrides,
            "distribution": {
                "mode": "downpour",
                "parameter_server_shards": 2,
                "mini_batch_targets": 10,
            },
        },
    ).with_seed(1)
    session_dp = create_session(
        spec_downpour.to_executor_config(),
        repo_root=Path(__file__).resolve().parents[3],
        cache_root=tmp_path / "cache",
    )
    assert isinstance(session_dp, DownpourTrainingSession)

    # Legacy config (table 2) produces TrainingSession
    spec_legacy = definition.load_run_spec(
        definition.config_root / "e01_table2_cbow.yaml",
        atomic_run_id="wmt--d50-w24m",
        overrides=base_overrides,
    ).with_seed(1)
    session_legacy = create_session(
        spec_legacy.to_executor_config(),
        repo_root=Path(__file__).resolve().parents[3],
        cache_root=tmp_path / "cache",
    )
    assert isinstance(session_legacy, TrainingSession)


@pytest.mark.parametrize(
    ("atomic_run_id", "model_kind"),
    (
        ("fineweb--cbow-d1000-w6000m", "cbow"),
        ("fineweb--skipgram-d1000-w6000m", "skip_gram"),
    ),
)
def test_table6_smoke_execution(
    tmp_path: Path, atomic_run_id: str, model_kind: str
) -> None:
    definition = DEFINITION.get_suite("w2v1")
    fixture_corpus = Path(__file__).parent / "fixtures/w2v1-corpus.txt"

    root = tmp_path / "repo"
    root.mkdir()
    paths = RuntimePaths(
        repo_root=Path(__file__).resolve().parents[3],
        data_root=root / "data",
        artifacts_root=root / "artifacts",
        cache_root=root / "cache",
        staging_root=root / "staging",
        references_root=root / "references",
        studies_root=root / "studies",
    )

    spec = definition.load_run_spec(
        definition.config_root / "e05_table6.yaml",
        atomic_run_id=atomic_run_id,
        overrides={
            "corpus": {"path": str(fixture_corpus)},
            "vocabulary": {
                "initial_capacity": 16,
                "hash_capacity": 128,
                "min_count": 1,
                "max_lexical_words": 100,
            },
            "training": {
                "embedding_dimension": 16,
                "epochs": 1,
                "thread_count": 2,
                "learning_rate_update_interval": 5,
                "observation_interval": 1,
            },
            "distribution": {
                "mode": "downpour",
                "parameter_server_shards": 2,
                "mini_batch_targets": 10,
                "adagrad_gamma": 0.05,
                "adagrad_epsilon": 1.0e-6,
                "queue_capacity": 16,
            },
        },
    ).with_seed(1)

    result = run_config(
        spec.to_executor_config(),
        ExperimentContext(paths=paths),
        executor_module=definition.executor_module,
    )

    # 1. Verify result report
    report = json.loads(result.report.read_text())
    assert report["complete"] is True
    assert len(report["epochs"]) == 1
    assert report["epochs"][0]["epoch"] == 1
    assert report["epochs"][0]["processed_tokens"] > 0

    # 2. Verify checkpoint
    ckpt = load_checkpoint(result.checkpoint)
    assert ckpt.completed_epochs == 1
    assert ckpt.processed_tokens == report["epochs"][0]["processed_tokens"]
    assert ckpt.input_embeddings().shape == (ckpt.vocabulary_state().size, 16)

    # 3. Verify lookup artifact
    lookup = load_lookup_artifact(result.lookup)
    assert lookup.embeddings.shape == (ckpt.vocabulary_state().size, 16)
