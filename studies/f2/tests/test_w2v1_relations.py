"""Checks for window distances, boundaries, deduplication and relation geometry."""

import hashlib

import numpy as np
import pytest

from f2.suites.w2v1.relations import context_counts, relation_statistics


def test_context_counts_filter_oov_and_do_not_cross_sentence_boundaries():
    tokens = [b"</s>", b"a", b"b", b"c", b"gap"]
    lines = [b"a missing c gap gap gap b\n", b"b\n", b"a\n"]
    counts, observed, provenance = context_counts(lines, tokens, [1, 2])
    # Retained distance(a,b)=5. OOV does not consume a window position.
    assert counts[:, 0, 2].tolist() == [0, 1, 6]
    assert counts[:, 1, 1].tolist() == [0, 1, 6]
    assert observed.tolist() == [2, 2]
    assert provenance["corpus_sha256"] == hashlib.sha256(b"".join(lines)).hexdigest()
    # Near token distance 1 gets probability 1 under dynamic radius 10.
    assert counts[:, 0, 3].tolist() == [1, 1, 10]


def test_relation_directions_are_scale_invariant_and_control_shared_endpoints():
    vectors = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 1.0]])
    pairs = [(0, 1), (2, 3), (2, 4)]
    stats, _, loo = relation_statistics(vectors, pairs)
    assert stats["mean_pairwise_cosine"] == pytest.approx(1)
    assert stats["mean_disjoint_pairwise_cosine"] == pytest.approx(1)
    assert stats["disjoint_comparisons"] == 2
    assert loo == pytest.approx([1, 1, 1])
    # Reverse one relation: directions disperse, while norms remain valid.
    reversed_stats, _, _ = relation_statistics(vectors, [(1, 0), (2, 3)])
    assert reversed_stats["mean_pairwise_cosine"] == pytest.approx(-1)
    assert reversed_stats["mean_direction_length"] == pytest.approx(0)


def test_zero_relations_are_reported_and_excluded_from_direction_means():
    stats, _, loo = relation_statistics(
        np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]), [(0, 1), (1, 2)]
    )
    assert stats["zero_relation_vectors"] == 1
    assert stats["mean_pairwise_cosine"] is None
    assert stats["mean_direction_length"] == 1
    assert np.isnan(loo).all()


def test_contexts_stop_at_literal_sentence_tokens_and_engine_block_limit():
    tokens = [b"</s>", b"a", b"b", b"gap"]
    counts, observed, _ = context_counts(
        [b"a </s> b\n", b"a " + b"gap " * 999 + b"b\n"], tokens, [1, 2]
    )
    assert counts[:, 0, 2].tolist() == [0, 0, 0]
    assert observed.tolist() == [2, 2]


def test_bound_corpus_bytes_match_budget_truncation_and_blank_lines(
    tmp_path, monkeypatch
):
    from f2.suites.w2v1 import relations

    payload = b"a b\n\nc d\n"

    class Store:
        def get_file(self, uri, path):
            path.write_bytes(payload)

    shards = [
        dict(
            shard_index=0,
            s3_uri="s3://fixture/shard",
            byte_size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    ]
    monkeypatch.setattr(
        relations,
        "_decompressed_lines",
        lambda path: iter(payload.splitlines(keepends=True)),
    )
    # The materializer preserves blank lines after an exactly exhausted line.
    assert (
        b"".join(relations.bound_corpus_lines(shards, Store(), tmp_path, 2))
        == b"a b\n\n"
    )
    assert (
        b"".join(relations.bound_corpus_lines(shards, Store(), tmp_path, 3))
        == b"a b\n\nc\n"
    )
    assert list(tmp_path.iterdir()) == []
