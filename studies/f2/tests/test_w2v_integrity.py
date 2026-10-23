"""Regression cases from the independent Phrase pipeline audit."""

from __future__ import annotations

import errno
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from f2.definition import DEFINITION
from f2.suites.w2v.phrases import (
    PhrasePolicy,
    cached_phrase_corpus,
    materialize_phrase_corpus,
)
from repro_core.context import ExperimentContext, RuntimePaths
from repro_core.execution.runner import run_config


def paths(root):
    return RuntimePaths(
        repo_root=root,
        data_root=root / "data",
        artifacts_root=root / "artifacts",
        cache_root=root / "cache",
        staging_root=root / "staging",
        references_root=root / "references",
        studies_root=root / "studies",
    )


def config(source, threshold=100, phrase_detection=None):
    definition = DEFINITION.get_suite("w2v2")
    return (
        definition.load_run_spec(
            definition.config_root / "e02_table3_phrase_skipgram.yaml",
            atomic_run_id="wmt--neg5-no-subsampling",
            overrides={
                "corpus": {
                    "path": str(source),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                },
                "phrase_detection": phrase_detection
                or {
                    "passes": 2,
                    "threshold": threshold,
                    "min_count": 5,
                },
                "training": {
                    "embedding_dimension": 4,
                    "epochs": 1,
                    "thread_count": 1,
                    "negative_table_size": 100,
                },
                "vocabulary": {
                    "min_count": 1,
                    "hash_capacity": 128,
                    "initial_capacity": 16,
                },
            },
        )
        .with_seed(1)
        .to_executor_config()
    )


def snapshot(root):
    return {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("change", ["policy", "source", "corrupt", "missing"])
def test_failed_resume_preserves_all_existing_results(tmp_path, change):
    source = tmp_path / "source"
    source.write_bytes(b"a b\n" * 6 + b"x\n" * 1800)
    settings = config(source)
    context = ExperimentContext(
        paths=paths(tmp_path), metadata={"run_root": tmp_path / "run"}
    )
    result = run_config(settings, context, executor_module="f2.suites.w2v2.executor")
    if change == "policy":
        settings = config(source, 25)
    elif change == "source":
        source.write_bytes(source.read_bytes() + b"c d\n")
        settings = config(source)
    elif change == "corrupt":
        (result.root / "checkpoints/latest.json").write_text("{")
    else:
        (result.root / "checkpoints/latest.json").unlink()
    before = snapshot(result.root)
    with pytest.raises((ValueError, RuntimeError)):
        run_config(settings, context, executor_module="f2.suites.w2v2.executor")
    assert snapshot(result.root) == before


def test_matching_rerun_and_unidentified_results_are_preserved(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"a b\n" * 10)
    settings = config(source)
    context = ExperimentContext(
        paths=paths(tmp_path), metadata={"run_root": tmp_path / "run"}
    )
    first = run_config(settings, context, executor_module="f2.suites.w2v2.executor")
    from f2.suites.w2v.artifacts import load_checkpoint

    expected = load_checkpoint(first.checkpoint).input_embeddings().copy()
    second = run_config(settings, context, executor_module="f2.suites.w2v2.executor")
    np.testing.assert_array_equal(
        load_checkpoint(second.checkpoint).input_embeddings(), expected
    )
    root = tmp_path / "unidentified"
    root.mkdir()
    (root / "valuable.txt").write_text("preserve")
    with pytest.raises(ValueError, match="preserved"):
        run_config(
            settings,
            ExperimentContext(paths=paths(tmp_path), metadata={"run_root": root}),
            executor_module="f2.suites.w2v2.executor",
        )
    assert (root / "valuable.txt").read_text() == "preserve"


@pytest.mark.parametrize(
    "mismatch",
    [
        "f2.config_digest",
        "f2.source_corpus_sha256",
        "seed",
        "suite.name",
        "implementation.variant",
        None,
    ],
)
def test_tracked_resume_requires_complete_identity(tmp_path, mismatch):
    from f2.suites.w2v.tracked import _find_resumable_run

    source = tmp_path / "source"
    source.write_bytes(b"a b\n" * 6)
    settings = config(source)
    tags = {
        "f2.config_digest": settings["identity"]["config_digest"],
        "f2.source_corpus_sha256": settings["corpus"]["sha256"],
        "seed": "1",
        "suite.name": "w2v2",
        "implementation.variant": settings["atomic_run_id"],
    }
    if mismatch:
        tags[mismatch] = "different"
    checkpoint = tmp_path / "staging/exp/f2/w2v2/tracked/old/checkpoints/latest.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}")
    restored = []
    client = SimpleNamespace(
        search_runs=lambda *a, **kw: [
            SimpleNamespace(
                data=SimpleNamespace(tags=tags),
                info=SimpleNamespace(run_id="old", lifecycle_stage="deleted"),
            )
        ],
        restore_run=restored.append,
    )
    actual = _find_resumable_run(
        client, "experiment", settings, paths(tmp_path), "w2v2"
    )
    assert actual == ("old" if mismatch is None else None)
    assert restored == (["old"] if mismatch is None else [])


@pytest.mark.parametrize("separator", [b"\v", b"\f", b"\0", b"a\vb"])
def test_incompatible_separator_is_rejected(separator):
    with pytest.raises(ValueError):
        PhrasePolicy(2, 0, 1, separator=separator)


@pytest.mark.parametrize(
    "data",
    [
        b"a\0b\n",
        b"</s>\n",
    ],
)
def test_unsupported_tokens_never_publish_or_reach_vocabulary(tmp_path, data):
    source = tmp_path / "source"
    source.write_bytes(data)
    with pytest.raises(ValueError, match="engine contract"):
        cached_phrase_corpus(source, tmp_path / "cache", PhrasePolicy(1, 0, 1))
    assert not list((tmp_path / "cache").rglob("phrases.txt"))


@pytest.mark.parametrize("size", [1, 2, 7, 64])
def test_chunk_boundary_preserves_documents_and_greedy_joins(
    tmp_path, monkeypatch, size
):
    import f2.suites.w2v.phrases as phrases

    monkeypatch.setattr(phrases, "TOKEN_CHUNK_BYTES", size)
    source = tmp_path / "source"
    source.write_bytes(b"\nalpha beta gamma\nalpha beta gamma\n\n tail  ")
    result = materialize_phrase_corpus(
        source, tmp_path / "output", PhrasePolicy(1, 0, 1)
    )
    assert result.path.read_bytes() == b"\nalpha_beta gamma\nalpha_beta gamma\n\ntail\n"
    assert result.token_count == 5


def test_long_line_memory_does_not_scale_with_document_length(tmp_path):
    import tracemalloc

    peaks = []
    for n in (2000, 200000):
        source = tmp_path / "source"
        with source.open("wb") as stream:
            for _ in range(n):
                stream.write(b"word01234567890 ")
        tracemalloc.start()
        result = materialize_phrase_corpus(
            source, tmp_path / "output", PhrasePolicy(1, 100, 1)
        )
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
        assert result.token_count == n
    assert peaks[1] < peaks[0] * 2 + 2_000_000


def test_injected_disk_full_never_publishes_and_retry_recovers(tmp_path, monkeypatch):
    import f2.suites.w2v.phrases as phrases

    source = tmp_path / "source"
    source.write_bytes(b"a b\na b\n")
    with monkeypatch.context() as patcher:

        def full(*args):
            raise OSError(errno.ENOSPC, "injected by test")

        patcher.setattr(phrases.os, "fsync", full)
        with pytest.raises(OSError):
            cached_phrase_corpus(source, tmp_path / "cache", PhrasePolicy(1, 0, 1))
    assert not list((tmp_path / "cache").rglob("phrases.txt"))
    assert (
        cached_phrase_corpus(
            source, tmp_path / "cache", PhrasePolicy(1, 0, 1)
        ).path.read_bytes()
        == b"a_b\na_b\n"
    )


def test_strict_evaluation_uses_verified_snapshot_even_when_file_changes(
    tmp_path, monkeypatch
):
    import f2.suites.w2v.evaluate as module

    source = tmp_path / "source"
    source.write_bytes(b"a b c d\n" * 6)
    settings = config(source)
    result = run_config(
        settings,
        ExperimentContext(
            paths=paths(tmp_path), metadata={"run_root": tmp_path / "run"}
        ),
        executor_module="f2.suites.w2v2.executor",
    )
    data = b": synthetic-audit\na b c d\n"
    questions = tmp_path / "questions"
    questions.write_bytes(data)

    # Synthetic identity is explicit and test-owned; no fixture is presented as official data.
    def verify(data):
        return {
            "questions_sha256": hashlib.sha256(data).hexdigest(),
            "total_questions": 1,
            "source": "test-only",
            "verified": True,
        }

    monkeypatch.setattr(module, "_phrase_questions_identity", verify)
    original = module.load_lookup_artifact

    def swap(path):
        questions.write_bytes(b": changed\nx y z q\nx y z q\n")
        return original(path)

    monkeypatch.setattr(module, "load_lookup_artifact", swap)
    report = module.evaluate_lookup_artifact("w2v2-table3", result.lookup, questions)
    assert (
        report["evaluation_identity"]["questions_sha256"]
        == hashlib.sha256(data).hexdigest()
    )
    assert report["analogy"]["overall"]["total_count"] == 1


def test_blocked_evaluation_matches_dense_reference_and_caps_allocations(monkeypatch):
    import f2.suites.w2v.evaluation as module
    from f2.suites.w2v.artifacts import LookupEmbeddings

    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(101, 12)).astype(np.float32)
    tokens = [f"w{i}".encode() for i in range(len(vectors))]
    packed = np.frombuffer(b"".join(tokens), dtype=np.uint8)
    offsets = np.array([0, *np.cumsum([len(t) for t in tokens])], dtype=np.uint64)
    lookup = LookupEmbeddings(vectors, packed, offsets, np.ones(101), {})
    rows = [(0, 1, 2, 3), (4, 5, 6, 7)]
    normalized = module._normalized_embeddings(vectors, dtype=np.float32)
    scores = (
        normalized @ (normalized[[1, 5]] - normalized[[0, 4]] + normalized[[2, 6]]).T
    )
    for c, question in enumerate(rows):
        scores[list(question[:3]), c] = -np.inf
    expected = list(
        zip(np.argmax(scores, axis=0), np.max(scores, axis=0) > 0, strict=True)
    )
    full = module._normalized_embeddings(vectors)
    nearest_scores = full @ full[0]
    nearest_order = sorted(
        range(1, len(vectors)), key=lambda i: (-float(nearest_scores[i]), tokens[i])
    )[:3]
    composition = full[[0, 1]].sum(axis=0)
    composition_scores = full @ (composition / np.linalg.norm(composition))
    composition_order = sorted(
        range(2, len(vectors)), key=lambda i: (-float(composition_scores[i]), tokens[i])
    )[:3]
    monkeypatch.setattr(module, "EVALUATION_CHUNK_ROWS", 7)
    original = module._normalized_embeddings
    seen = []

    def bounded(values, **kwargs):
        seen.append(len(values))
        assert len(values) <= 7
        return original(values, **kwargs)

    monkeypatch.setattr(module, "_normalized_embeddings", bounded)
    assert list(module._analogy_predictions(vectors, rows, 2)) == expected
    nearest = module.nearest_tokens(lookup, b"w0", limit=3)
    composed = module.additive_composition(lookup, [b"w0", b"w1"], limit=3)
    assert [item.token for item in nearest] == [tokens[i] for i in nearest_order]
    assert [item.token for item in composed] == [tokens[i] for i in composition_order]
    np.testing.assert_allclose(
        [item.score for item in nearest], nearest_scores[nearest_order]
    )
    np.testing.assert_allclose(
        [item.score for item in composed], composition_scores[composition_order]
    )
    assert seen and lookup._index is not None
    assert lookup._index.execute("SELECT count(*) FROM rows").fetchone()[0] == 101


def test_candidate_exclusion_oov_denominator_and_ties_across_blocks(monkeypatch):
    import f2.suites.w2v.evaluation as module

    lookup = SimpleNamespace(
        tokens=(b"a", b"b", b"c", b"d", b"e"),
        embeddings=np.array([[1, 0], [0, 1], [1, 0], [0, 1], [0, 1]], dtype=np.float32),
    )
    monkeypatch.setattr(module, "EVALUATION_CHUNK_ROWS", 2)
    result = module.evaluate_analogies(
        lookup,
        module.parse_analogy_questions([b": test", b"a b c d", b"a b missing d"]),
    )
    assert (
        result.overall.score,
        result.overall.total_count,
        result.overall.valid_count,
    ) == (1.0, 2, 1)


def test_supported_99_byte_phrases_remain_distinct_in_engine(tmp_path):
    from w2v import Corpus, Vocabulary, VocabularyConfig

    left = b"a" * 49
    first = b"b" * 49
    second = b"b" * 48 + b"c"
    source = tmp_path / "source"
    source.write_bytes(
        (left + b" " + first + b"\n") * 2 + (left + b" " + second + b"\n") * 2
    )
    phrase = materialize_phrase_corpus(
        source, tmp_path / "output", PhrasePolicy(1, 0, 1)
    )
    vocabulary = Vocabulary.build(
        Corpus(phrase.path),
        VocabularyConfig(initial_capacity=16, hash_capacity=128, min_count=1),
    )
    packed = vocabulary.token_bytes()
    offsets = vocabulary.token_offsets()
    tokens = {
        bytes(packed[offsets[i] : offsets[i + 1]]): int(vocabulary.counts()[i])
        for i in range(vocabulary.size)
    }
    assert tokens[left + b"_" + first] == tokens[left + b"_" + second] == 2


def test_mutating_input_during_construction_never_publishes(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"a b\na b\n")

    def mutate(message):
        if message.endswith("counting"):
            source.write_bytes(b"c d\nc d\n")

    with pytest.raises(ValueError, match="changed"):
        cached_phrase_corpus(
            source, tmp_path / "cache", PhrasePolicy(1, 0, 1), progress=mutate
        )
    assert not list((tmp_path / "cache").rglob("phrases.txt"))


def test_phrase_checkpoint_without_provable_run_identity_is_preserved(tmp_path):
    from f2.suites.w2v.artifacts import create_checkpoint_manager
    from f2.suites.w2v.executor import create_session

    source = tmp_path / "source"
    source.write_bytes(b"a b\n" * 6)
    settings = config(source)
    session = create_session(settings, tmp_path, cache_root=tmp_path / "cache")
    session.train_epoch()
    root = tmp_path / "old-run"
    create_checkpoint_manager(root / "checkpoints", session=session).save_latest()
    before = snapshot(root)
    with pytest.raises(ValueError, match="missing phrase run identity"):
        run_config(
            settings,
            ExperimentContext(paths=paths(tmp_path), metadata={"run_root": root}),
            executor_module="f2.suites.w2v2.executor",
        )
    assert snapshot(root) == before


def test_long_phrases_train_restore_and_evaluate_without_aliases(tmp_path):
    from f2.suites.w2v.artifacts import load_checkpoint, load_lookup_artifact
    from f2.suites.w2v.evaluate import evaluate_lookup_artifact

    left = b"a" * 60
    right = b"b" * 60
    other = b"b" * 59 + b"c"
    source = tmp_path / "source"
    # Longer than both the legacy tokenizer limit and the phrase chunk size.
    giant = b"g" * 70000
    source.write_bytes(
        ((left + b" " + right + b" " + left + b" " + other + b"\n") * 6) + giant + b"\n"
    )
    settings = config(
        source, 0, phrase_detection={"passes": 1, "threshold": 0, "min_count": 1}
    )
    context = ExperimentContext(
        paths=paths(tmp_path), metadata={"run_root": tmp_path / "run"}
    )
    first = run_config(settings, context, executor_module="f2.suites.w2v2.executor")
    lookup = load_lookup_artifact(first.lookup)
    assert lookup.row(left + b"_" + right) != lookup.row(left + b"_" + other)
    assert lookup.row(giant) is not None
    expected = load_checkpoint(first.checkpoint).input_embeddings().copy()
    rerun = run_config(settings, context, executor_module="f2.suites.w2v2.executor")
    np.testing.assert_array_equal(
        load_checkpoint(rerun.checkpoint).input_embeddings(), expected
    )
    questions = tmp_path / "questions"
    questions.write_bytes(
        b": explicit-partial-fixture\n"
        + left
        + b"_"
        + right
        + b" "
        + left
        + b"_"
        + other
        + b" "
        + giant
        + b" "
        + left
        + b"_"
        + right
        + b"\n"
    )
    result = evaluate_lookup_artifact("w2v2", first.lookup, questions)
    assert result["analogy"]["overall"]["valid_count"] == 1


def test_vocabulary_cache_separates_tokenization_modes(tmp_path):
    from w2v import Corpus

    from f2.suites.w2v.artifacts import resolve_or_build_vocabulary

    source = tmp_path / "source"
    source.write_bytes((b"a" * 120 + b"x\n") * 2 + (b"a" * 120 + b"y\n") * 2)
    values = {"initial_capacity": 16, "hash_capacity": 128, "min_count": 1}
    old = resolve_or_build_vocabulary(
        Corpus(source), values, cache_root=tmp_path / "cache"
    )
    exact = resolve_or_build_vocabulary(
        Corpus(source, full_tokens=True), values, cache_root=tmp_path / "cache"
    )
    assert old.size == 2 and exact.size == 3
    assert (
        len(list((tmp_path / "cache/f2/w2v/vocabulary").glob("*/manifest.json"))) == 2
    )
