"""Corpus-side accounting tests with boundaries, duplicates and OOV gaps."""

from collections import Counter

import numpy as np

from f2.suites.w2v1.corpus_investigation import bound_lines, prediction_proxy
from f2.suites.w2v1.corpus_statistics import (
    BottomHashes,
    CorpusStatistics,
    js_divergence,
    spelling_key,
    training_blocks,
)


def test_streaming_counts_reproduce_oov_distances_and_duplicate_exposure(tmp_path):
    tokens = [b"</s>", b"Paris", b"France", b"news"]
    pairs = [dict(a=1, b=2, kind="relation", category="capital-world")]
    stats = CorpusStatistics(tokens, np.array([3, 2, 2, 1]), [1, 2], pairs, tmp_path, 1)
    stats.consume(b"Paris missing France\n", 0)
    stats.consume(b"Paris missing France\n", 0)
    stats.consume(b"news\n", 0)
    result = stats.finish(tmp_path)
    np.testing.assert_array_equal(stats.observed, [3, 2, 2, 1])
    assert stats.word[0, 1] == 2  # Two sentence instances.
    assert stats.word[0, 2] == 1  # One deduplicated occurrence.
    assert stats.word[0, 4] == 2
    assert stats.word[0, 7] == 2  # Both distances changed after removing OOV.
    assert stats.word[0, 10] == 1.8  # Raw distance two, probability 9/10.
    assert stats.contexts[0, 1, 0, 2] == 2
    assert stats.contexts[1, 1, 0, 2] == 20  # Filtered distance one.
    assert result["total"]["oov_bridged_adjacencies"] == 2
    assert stats.patterns[0, 0] == Counter({"a_left|": 2})
    assert result["unique_contexts"][0] == (1.0, True)


def test_sentence_and_engine_block_boundaries_are_distinct():
    vocabulary = {b"</s>": 0, b"a": 1, b"b": 2}
    assert list(training_blocks([b"a", b"</s>", b"b"], vocabulary)) == [
        [(1, 0)],
        [(2, 2)],
    ]
    blocks = list(training_blocks([b"a"] * 1001, vocabulary))
    assert [len(block) for block in blocks] == [1000, 1]
    assert spelling_key(b'"NEW-YORK,"') == b"newyork"


def test_cardinality_sketch_is_exact_before_capacity_and_deduplicates():
    sketch = BottomHashes(k=4)
    for value in (10, 10, 20, 30):
        sketch.add(value)
    assert sketch.estimate() == (3.0, True)
    for value in range(100):
        sketch.add(value)
    assert len(sketch.values) == 4
    assert sketch.values == {0, 1, 2, 3}
    assert sketch.estimate()[1] is False


def test_js_divergence_extremes():
    assert js_divergence(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 0
    assert js_divergence(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 1


def test_grouped_proxy_uses_only_training_templates_and_reports_chance():
    from f2.suites.w2v1.corpus_statistics import hash64

    signatures = [i.to_bytes(16, "little").hex() for i in range(20)]
    train = next(s for s in signatures if hash64(bytes.fromhex(s)) % 5 != 0)
    test = next(s for s in signatures if hash64(bytes.fromhex(s)) % 5 == 0)
    samples = [
        [(1, train, [3], [3]), (2, test, [3], [3])],
        [(1, train, [4], [4]), (2, test, [4], [4])],
    ]
    rows = prediction_proxy(samples, {"family": {1, 2}}, [1, 2], "fixture")
    assert len(rows) == 4
    assert all(r["heldout_accuracy"] == 1 and r["chance_accuracy"] == 0.5 for r in rows)


def test_annotated_materialization_keeps_exact_partial_line_and_cleans_shard(
    tmp_path, monkeypatch
):
    from f2.suites.w2v1 import corpus_investigation as module

    class Store:
        def get_file(self, uri, path):
            path.write_bytes(b"compressed")

    monkeypatch.setattr(module, "sha256_file", lambda path: "checksum")
    monkeypatch.setattr(
        module, "_decompressed_lines", lambda path: iter([b"a b\n", b"\n", b"c d e\n"])
    )
    shards = [
        dict(
            shard_index=0, s3_uri="s3://fixture/shard", byte_size=10, sha256="checksum"
        )
    ]
    assert list(bound_lines(shards, Store(), tmp_path, budget=3)) == [
        (0, b"a b\n"),
        (0, b"\n"),
        (0, b"c\n"),
    ]
    assert not list(tmp_path.iterdir())


def test_count_based_center_information_separates_disjoint_from_shared_contexts(
    tmp_path,
):
    from f2.suites.w2v1.corpus_evidence import discriminability_evidence

    contexts = np.lib.format.open_memmap(
        tmp_path / "contexts.npy", mode="w+", dtype=np.uint32, shape=(2, 2, 4, 5)
    )
    contexts[:, :, 0, 3] = 10
    contexts[:, :, 1, 4] = 10
    contexts[:, :, 2, 3] = 10
    contexts[:, :, 3, 3] = 10
    config = {
        "targets": [1, 2, 3, 4],
        "controls": {"1": 3, "2": 4},
        "groups": {"family": [1, 2]},
    }
    words = [
        dict(row=r, word=str(r), window=w, context_events=20)
        for r in range(1, 5)
        for w in ("fixed4", "dynamic10_expected")
    ]
    relations = [
        dict(category="family", window=w, word_a="1", word_b="2")
        for w in ("fixed4", "dynamic10_expected")
    ]
    records = discriminability_evidence(
        "fixture", config, tmp_path, tmp_path, relations, words, []
    )
    assert all(
        r["single_context_center_information_bits"] == 1
        for r in records
        if r["category"] == "family"
    )
    assert all(
        r["single_context_center_information_bits"] == 0
        for r in records
        if r["category"] == "family-matched-controls"
    )
    assert all(
        r["single_context_correct_center_probability"] == 0.5
        for r in records
        if r["category"] == "family-matched-controls"
    )


def test_accuracy_aggregates_keep_common_question_subset_and_unique_pairs():
    from types import SimpleNamespace

    from f2.suites.w2v1.corpus_report import accuracy_records

    questions = [SimpleNamespace(category="family"), SimpleNamespace(category="family")]
    data = {
        "mapped": [(1, 2, 3, 4), (5, 6, 3, 4)],
        "tokens": [str(i).encode() for i in range(7)],
    }
    outcomes = [
        dict(
            question_id=0,
            model="cbow",
            seed=1,
            included=True,
            common_subset=True,
            correct=True,
        ),
        dict(
            question_id=1,
            model="cbow",
            seed=1,
            included=True,
            common_subset=False,
            correct=False,
        ),
    ]
    category, pairs, words = accuracy_records("fixture", data, questions, outcomes)
    assert len(pairs) == 1
    assert pairs[0]["accuracy_percent"] == 50
    assert pairs[0]["common_accuracy_percent"] == 100
    total = next(r for r in category if r["category"] == "total")
    assert total["accuracy_percent"] == 50 and total["common_accuracy_percent"] == 100
    assert next(r for r in words if r["row"] == 4)["question_count"] == 2


def test_buffered_contexts_do_not_cross_flushes_or_sentences(tmp_path):
    tokens = [b"</s>", b"a", b"b"]
    stats = CorpusStatistics(tokens, np.array([3, 2, 1]), [1], [], tmp_path, 1)
    stats.consume(b"a\n", 0)
    stats.flush()
    stats.consume(b"a b\n", 0)
    stats.consume(b"\n", 0)
    stats.finish(tmp_path)
    assert stats.contexts[0, 1, 0, 2] == 1
    assert stats.word[0, 8] == 1
    np.testing.assert_array_equal(stats.observed, [3, 2, 1])
