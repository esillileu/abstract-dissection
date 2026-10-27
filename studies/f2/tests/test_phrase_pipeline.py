"""Exact phrase detection, bounded aggregation, and immutable cache lifecycle."""

from __future__ import annotations

import json
import multiprocessing
import os
import tracemalloc
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path

import pytest

from f2.suites.w2v.phrases import (
    PhrasePolicy,
    cached_phrase_corpus,
    materialize_phrase_corpus,
)


def _reference(data, policy):
    for threshold in policy.thresholds or (policy.threshold,) * policy.passes:
        documents = [line.split() for line in data.splitlines()]
        words = Counter(t for doc in documents for t in doc)
        pairs = Counter(pair for doc in documents for pair in pairwise(doc))
        output = []
        for doc in documents:
            result = []
            i = 0
            while i < len(doc):
                if i + 1 < len(doc):
                    a, b = doc[i : i + 2]
                    n = pairs[a, b]
                    if (
                        n >= policy.min_count
                        and (n - policy.min_count)
                        * sum(words.values())
                        / (words[a] * words[b])
                        > threshold
                    ):
                        result.append(a + policy.separator + b)
                        i += 2
                        continue
                result.append(doc[i])
                i += 1
            output.append(b" ".join(result) + b"\n")
        data = b"".join(output)
    return data


@pytest.mark.parametrize("passes,scheduled", [(1, False), (2, True), (3, False)])
def test_disk_counts_equal_historical_counts(tmp_path, passes, scheduled):
    source = tmp_path / "input"
    data = b"new york city new york city\n\xff x \xff x\n\nyork\nnew\n" * 20
    source.write_bytes(data)
    policy = PhrasePolicy(
        passes,
        1.0,
        1,
        thresholds=tuple(4 / (i + 1) for i in range(passes)) if scheduled else None,
    )
    result = materialize_phrase_corpus(source, tmp_path / "output", policy)
    assert result.path.read_bytes() == _reference(data, policy)
    lineage = json.loads(result.path.with_suffix(".json").read_text())
    assert len(lineage["pass_statistics"]) == passes
    assert lineage["pass_statistics"][0]["input_tokens"] == len(data.split())
    assert lineage["token_count"] + lineage["joined_count"] == len(data.split())
    assert result.path.read_bytes().splitlines()[-2:] == [b"york", b"new"]


def _build(arguments):
    source, root = arguments
    return cached_phrase_corpus(source, root, PhrasePolicy(2, 0, 1))


def test_concurrent_build_reuse_and_corruption_recovery(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.write_bytes(b"new york city new york city\n" * 5)
    with ProcessPoolExecutor(
        2,
        mp_context=multiprocessing.get_context(
            "fork" if hasattr(os, "fork") else "spawn"
        ),
    ) as workers:
        results = list(workers.map(_build, [(source, tmp_path / "cache")] * 2))
    assert len({r.path for r in results}) == 1
    first = results[0]
    stamp = first.path.stat().st_mtime_ns
    import f2.suites.w2v.phrases as phrases

    original = phrases.materialize_phrase_corpus

    def fail(*args, **kwargs):
        pytest.fail("valid cache rebuilt")

    monkeypatch.setattr(phrases, "materialize_phrase_corpus", fail)
    assert _build((source, tmp_path / "cache")) == first
    assert first.path.stat().st_mtime_ns == stamp
    monkeypatch.setattr(phrases, "materialize_phrase_corpus", original)
    for target, data in [
        (first.path, b"corrupt"),
        (first.path.with_suffix(".txt.json"), b"[]"),
    ]:
        target.write_bytes(data)
        rebuilt = _build((source, tmp_path / "cache"))
        assert rebuilt == first
    changed = cached_phrase_corpus(source, tmp_path / "cache", PhrasePolicy(2, 100, 1))
    assert changed.path != first.path
    with pytest.raises(ValueError, match="checksum"):
        cached_phrase_corpus(
            source, tmp_path / "cache", PhrasePolicy(2, 0, 1), source_sha256="bad"
        )


def _die_during_pass(source, root):
    def die(message):
        if "pass=2/2 counting" in message:
            os._exit(17)

    cached_phrase_corpus(source, root, PhrasePolicy(2, 0, 1), progress=die)


def test_killed_builder_restarts_without_partial_publication(tmp_path):
    source = tmp_path / "input"
    source.write_bytes(b"new york city new york city\n" * 5)
    root = tmp_path / "cache"
    ctx = multiprocessing.get_context("fork" if hasattr(os, "fork") else "spawn")
    process = ctx.Process(target=_die_during_pass, args=(source, root))
    process.start()
    process.join(20)
    assert process.exitcode == 17
    assert not list(root.rglob("phrases.txt.json"))
    rebuilt = _build((source, root))
    assert rebuilt.path.is_file()
    assert not list((root / "f2/w2v/phrases").glob(".*"))
    expected = materialize_phrase_corpus(
        source, tmp_path / "expected", PhrasePolicy(2, 0, 1)
    )
    assert rebuilt.corpus_sha256 == expected.corpus_sha256


def test_unique_bigram_growth_does_not_grow_python_count_tables(tmp_path):
    peaks = []
    for count in (500, 2500):
        source = tmp_path / f"input-{count}"
        with source.open("wb") as stream:
            for i in range(count):
                stream.write(f"a{i} b{i} c{i}\n".encode())
        tracemalloc.start()
        result = materialize_phrase_corpus(
            source, tmp_path / f"output-{count}", PhrasePolicy(1, 0, 1)
        )
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
        assert result.token_count == count * 3
    # Fixed streaming/hash buffers dominate; sqlite pages have a separate 16 MiB cap.
    assert peaks[1] < peaks[0] * 2 + 500_000


@pytest.mark.parametrize("thresholds", [(1,), (float("nan"), 1), (-1, 0)])
def test_invalid_threshold_schedule(thresholds):
    with pytest.raises(ValueError):
        PhrasePolicy(2, 1, 1, thresholds=thresholds)


def test_full_benchmark_does_not_accept_fixture():
    from f2.suites.w2v.evaluate import verify_phrase_questions

    with pytest.raises(ValueError, match="3218"):
        verify_phrase_questions(
            Path(__file__).parent / "fixtures/questions-phrases.txt"
        )
