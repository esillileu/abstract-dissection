from __future__ import annotations

import hashlib

import numpy as np
import pytest

from f2.definition import DEFINITION
from f2.suites.w2v.artifacts import (
    load_checkpoint,
    load_lookup_artifact,
)
from f2.suites.w2v.evaluation import (
    additive_composition,
    nearest_tokens,
    pca_projection,
)
from f2.suites.w2v.phrases import PhrasePolicy, materialize_phrase_corpus
from f2.suites.w2v1.executor import restore_session
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


def test_w2v2_local_phrase_run_resume_lookup_and_reports(tmp_path):
    definition = DEFINITION.get_suite("w2v2")
    assert definition.executor_module == "f2.suites.w2v2.executor"
    spec = definition.load_run_spec(
        definition.config_root / "e01_phrase_skipgram.yaml",
        atomic_run_id="local-smoke",
        overrides={},
    )
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
    assert (result.root / "phrase_evaluation.json").is_file()
    lookup = load_lookup_artifact(result.lookup)
    assert lookup.row(b"new_york") is not None
    assert nearest_tokens(lookup, b"new_york", limit=2)
    assert additive_composition(lookup, [b"new_york", b"city"], limit=2)
    projection = pca_projection(lookup, [b"new_york", b"city_san", b"francisco"])
    assert set(projection) == {b"new_york", b"city_san", b"francisco"}

    interrupted_config = dict(config)
    phrase_path = (
        paths.staging_root
        / "exp/f2/w2v2/phrase-corpus/derived/w2v2-local-smoke-s1/phrases.txt"
    )
    interrupted_config["corpus"] = {
        "path": str(phrase_path),
        "sha256": hashlib.sha256(phrase_path.read_bytes()).hexdigest(),
    }
    session = restore_session(interrupted_config, result.checkpoint, paths.repo_root)
    np.testing.assert_array_equal(
        session.export_state().input_embeddings(), state.input_embeddings()
    )


def test_w2v2_rejects_nce_substitution():
    definition = DEFINITION.get_suite("w2v2")
    with pytest.raises(ValueError, match="NCE is not substituted"):
        definition.load_run_spec(
            definition.config_root / "e01_phrase_skipgram.yaml",
            atomic_run_id="local-smoke",
            overrides={"training": {"objective_kind": "nce"}},
        )
