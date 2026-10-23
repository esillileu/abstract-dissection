from __future__ import annotations

from pathlib import Path

import pytest

from f2.definition import DEFINITION
from f2.suites.w2v.artifacts import (
    load_checkpoint,
    load_lookup_artifact,
)
from f2.suites.w2v.evaluate import evaluate_lookup_artifact
from f2.suites.w2v.evaluation import (
    additive_composition,
    nearest_tokens,
    pca_projection,
)
from f2.suites.w2v.phrases import PhrasePolicy, materialize_phrase_corpus
from repro_core.context import ExperimentContext, RuntimePaths
from repro_core.execution.runner import run_config


def _paths(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    return RuntimePaths(
        repo_root=DEFINITION.get_suite("w2v2").config_root.parents[6],
        data_root=root / "data",
        artifacts_root=root / "artifacts",
        cache_root=root / "cache",
        staging_root=root / "staging",
        references_root=root / "references",
        studies_root=root / "studies",
    )


def test_phrase_materialization_is_deterministic_and_byte_preserving(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"new york city new york city\n\xff x \xff x\n")
    policy = PhrasePolicy(passes=1, threshold=0.0, min_count=1)
    first = materialize_phrase_corpus(source, tmp_path / "first.txt", policy)
    second = materialize_phrase_corpus(source, tmp_path / "second.txt", policy)
    assert first.corpus_sha256 == second.corpus_sha256
    assert first.lineage_digest == second.lineage_digest
    assert b"new_york" in first.path.read_bytes()
    assert b"\xff_x" in first.path.read_bytes()


def test_phrase_materialization_reports_each_streaming_pass(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"new york city\nnew york state\n")
    messages: list[str] = []
    materialize_phrase_corpus(
        source,
        tmp_path / "phrases.txt",
        PhrasePolicy(passes=2, threshold=0.0, min_count=1),
        progress=messages.append,
    )
    assert messages[0] == "phrase pass=1/2 counting"
    assert any("phrase pass=1/2 complete" in message for message in messages)
    assert any("phrase pass=2/2 complete" in message for message in messages)


def test_w2v2_phrase_run_completes_declared_schedule_and_reports(tmp_path):
    definition = DEFINITION.get_suite("w2v2")
    assert definition.executor_module == "f2.suites.w2v2.executor"
    spec = definition.load_run_spec(
        definition.config_root / "e02_table3_phrase_skipgram.yaml",
        atomic_run_id="wmt--neg5-subsampling",
        overrides={
            "corpus": {"path": str(Path(__file__).parent / "fixtures/w2v2-corpus.txt")},
            "vocabulary": {
                "initial_capacity": 16,
                "hash_capacity": 128,
                "min_count": 1,
            },
            "training": {
                "embedding_dimension": 8,
                "window_radius": 2,
                "epochs": 2,
                "thread_count": 1,
                "negative_table_size": 100,
            },
            "phrase_detection": {"passes": 1, "threshold": 0.0, "min_count": 1},
        },
    ).with_seed(1)
    config = spec.to_executor_config()
    paths = _paths(tmp_path)
    result = run_config(
        config,
        ExperimentContext(paths=paths),
        executor_module=definition.executor_module,
    )
    state = load_checkpoint(result.checkpoint)
    assert state.completed_epochs == 2
    assert (result.root / "phrase_lineage.json").is_file()
    assert not (result.root / "phrase_evaluation.json").exists()
    lookup = load_lookup_artifact(result.lookup)
    assert lookup.row(b"new_york") is not None
    assert nearest_tokens(lookup, b"new_york", limit=2)
    assert additive_composition(lookup, [b"new_york", b"city"], limit=2)
    projection = pca_projection(lookup, [b"new_york", b"city_san", b"francisco"])
    assert set(projection) == {b"new_york", b"city_san", b"francisco"}
    evaluation = evaluate_lookup_artifact(
        "w2v2",
        result.lookup,
        Path(__file__).parent / "fixtures/questions-phrases.txt",
    )
    assert evaluation["analogy"]["overall"]["total_count"] > 0
    assert evaluation["phrase"]["nearest_entities"]
    assert evaluation["phrase"]["additive_composition"]["tokens"]
    assert evaluation["phrase"]["pca"]


def test_w2v2_rejects_nce_substitution():
    definition = DEFINITION.get_suite("w2v2")
    with pytest.raises(ValueError, match="NCE is not substituted"):
        definition.load_run_spec(
            definition.config_root / "e02_table3_phrase_skipgram.yaml",
            atomic_run_id="wmt--neg5-subsampling",
            overrides={"training": {"objective_kind": "nce"}},
        )


def test_canonical_phrase_training_uses_twenty_workers() -> None:
    definition = DEFINITION.get_suite("w2v2")
    config = definition.load_run_spec(
        definition.config_root / "e02_table3_phrase_skipgram.yaml",
        atomic_run_id="wmt--neg5-subsampling",
        overrides={},
    ).to_executor_config()

    assert config["training"]["thread_count"] == 20


def test_evaluation_resources_do_not_change_w2v2_training_identity() -> None:
    definition = DEFINITION.get_suite("w2v2")
    source = definition.config_root / "e02_table3_phrase_skipgram.yaml"
    first = definition.load_run_spec(
        source,
        atomic_run_id="wmt--neg5-subsampling",
        overrides={"evaluation": {"questions_path": "first.txt"}},
    ).to_executor_config()
    second = definition.load_run_spec(
        source,
        atomic_run_id="wmt--neg5-subsampling",
        overrides={"evaluation": {"questions_path": "second.txt"}},
    ).to_executor_config()
    assert "evaluation" not in first
    assert first["identity"]["config_digest"] == second["identity"]["config_digest"]


TABLE1_CONDITIONS = (
    "neg5-no-subsampling",
    "neg15-no-subsampling",
    "hs-no-subsampling",
    "neg5-subsampling",
    "neg15-subsampling",
    "hs-subsampling",
)


def test_table1_plans_and_catalog_slots_are_distinct_and_bound():
    from f2.catalog.manifest import read_manifest
    from repro_core.execution.definition import RunOptions, RunSelection
    from repro_core.execution.planning import Planner

    definition = DEFINITION.get_suite("w2v2")
    plans = Planner(definition).build(
        RunSelection(experiment_ids=("e01",)), RunOptions()
    )
    assert len(plans) == 54
    manifest = read_manifest(Path(__file__).parents[1] / "catalog/w2v.json")
    slots = {row["planned_run_slot_id"]: row for row in manifest["planned_run_slots"]}
    seen = set()
    for plan in plans:
        spec = definition.load_run_spec(
            plan.path, atomic_run_id=plan.atomic_run_id, overrides={}
        ).with_seed(plan.seed)
        config = spec.to_executor_config()
        slot = str(config["identity"]["planned_run_slot_id"])
        assert slot not in seen
        seen.add(slot)
        assert slots[slot]["seed"] == plan.seed
        assert slots[slot]["parameters"]["threads"] == 20
        assert config["corpus_mode"] == "word"
        assert "phrase_detection" not in config
        assert config["training"]["embedding_dimension"] == 300
        assert config["vocabulary"]["min_count"] == 5
        assert config["vocabulary"].get("max_lexical_words", 0) == 0
        corpus, condition = plan.atomic_run_id.split("--")
        assert condition in TABLE1_CONDITIONS
        assert config["corpus"]["lexical_token_budget"] == (
            791_844_834 if corpus == "lm1b" else 1_000_000_000
        )
        assert config["training"]["subsampling_threshold"] == (
            0.0 if condition.endswith("no-subsampling") else 1e-5
        )
        objective = config["training"]["objective_kind"]
        assert objective == (
            "hierarchical_softmax"
            if condition.startswith("hs")
            else "negative_sampling"
        )
        assert config["training"]["negative_sample_count"] == (
            0
            if condition.startswith("hs")
            else 15
            if condition.startswith("neg15")
            else 5
        )


@pytest.mark.parametrize("condition", TABLE1_CONDITIONS)
def test_table1_word_training_restore_and_rerun(tmp_path, monkeypatch, condition):
    import numpy as np

    from f2.suites.w2v import executor as shared_executor
    from f2.suites.w2v.artifacts import create_checkpoint_manager
    from f2.suites.w2v.executor import create_session

    definition = DEFINITION.get_suite("w2v2")
    source = tmp_path / "words.txt"
    # Long sentences ensure real context pairs survive the canonical 1e-5 threshold.
    sentence = b"king queen man woman paris france berlin germany " * 100 + b"\n"
    source.write_bytes(sentence * 100)
    spec = definition.load_run_spec(
        definition.config_root / "e01_table1_word_skipgram.yaml",
        atomic_run_id=f"wmt--{condition}",
        overrides={
            "corpus": {"path": str(source)},
            "vocabulary": {"initial_capacity": 16, "hash_capacity": 128},
            "training": {
                "embedding_dimension": 8,
                "epochs": 2,
                "thread_count": 1,
                "negative_table_size": 1000,
                "observation_interval": 1000,
            },
        },
    ).with_seed(7)
    config = spec.to_executor_config()
    paths = _paths(tmp_path)

    def no_phrases(*args, **kwargs):
        pytest.fail("word-only training attempted phrase materialization")

    monkeypatch.setattr("f2.suites.w2v2.executor.materialize_phrase_corpus", no_phrases)
    baseline = run_config(
        config,
        ExperimentContext(paths=paths, metadata={"run_root": tmp_path / "baseline"}),
        executor_module=definition.executor_module,
    )
    expected = load_checkpoint(baseline.checkpoint)
    assert expected.completed_epochs == 2
    assert np.any(expected.output_embeddings() != 0)
    assert np.isfinite(expected.input_embeddings()).all()
    lookup = load_lookup_artifact(baseline.lookup)
    assert lookup.row(b"king") is not None
    assert nearest_tokens(lookup, b"king", limit=2)
    questions = tmp_path / "questions-words.txt"
    questions.write_bytes(b": capital-common-countries\nparis france berlin germany\n")
    evaluation = evaluate_lookup_artifact("w2v2", baseline.lookup, questions)
    assert evaluation["analogy"]["overall"]["total_count"] == 1
    assert evaluation["analogy"]["overall"]["valid_count"] == 1
    assert not list(paths.staging_root.rglob("phrases*"))
    assert not (baseline.root / "phrase_lineage.json").exists()

    # Build an epoch-boundary checkpoint through the production adapter, then
    # let the ordinary executor restore and finish the remaining schedule.
    session = create_session(config, paths.repo_root, cache_root=paths.cache_root)
    session.train_epoch()
    resumed_root = tmp_path / "resumed"
    manager = create_checkpoint_manager(resumed_root / "checkpoints", session=session)
    manager.save_latest()

    def no_restart(*args, **kwargs):
        pytest.fail("checkpoint was ignored and training restarted")

    monkeypatch.setattr(shared_executor, "create_session", no_restart)
    context = ExperimentContext(paths=paths, metadata={"run_root": resumed_root})
    resumed = run_config(config, context, executor_module=definition.executor_module)
    restored = load_checkpoint(resumed.checkpoint)
    np.testing.assert_array_equal(
        expected.input_embeddings(), restored.input_embeddings()
    )
    np.testing.assert_array_equal(
        expected.output_embeddings(), restored.output_embeddings()
    )
    rerun = run_config(config, context, executor_module=definition.executor_module)
    np.testing.assert_array_equal(
        expected.input_embeddings(),
        load_checkpoint(rerun.checkpoint).input_embeddings(),
    )
    assert not (resumed.root / "phrase_lineage.json").exists()
    # All objectives and the resumed execution reuse a single word vocabulary.
    assert (
        len(list((paths.cache_root / "f2/w2v/vocabulary").glob("*/manifest.json"))) == 1
    )


@pytest.mark.parametrize(
    "override, message",
    [
        ({"corpus_mode": "unknown"}, "corpus_mode"),
        ({"subsampling_policy": "paper"}, "subsampling_policy"),
        ({"subsampling_policy": None}, "subsampling_policy"),
        ({"phrase_detection": {}}, "phrase_detection"),
        ({"training": {"objective_kind": "nce"}}, "NCE is not substituted"),
    ],
)
def test_table1_rejects_unsupported_policies(override, message):
    definition = DEFINITION.get_suite("w2v2")
    with pytest.raises(ValueError, match=message):
        definition.load_run_spec(
            definition.config_root / "e01_table1_word_skipgram.yaml",
            atomic_run_id="wmt--neg5-subsampling",
            overrides=override,
        )
