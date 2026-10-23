"""Independent byte-preservation and memory regressions for corpus preparation."""

from __future__ import annotations

import io
import random
import re
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from f2.suites.w2v.corpus import _copy_corpus_prefix


def reference(data, budget):
    out = io.BytesIO()
    tokens = 0
    for line in io.BytesIO(data):
        matches = list(re.finditer(rb"[^ \t\r\n]+", line))
        available = budget - tokens
        if len(matches) <= available:
            out.write(line)
            tokens += len(matches)
        else:
            if available:
                out.write(line[: matches[available - 1].end()] + b"\n")
                tokens += available
            return out.getvalue(), tokens, False
    return out.getvalue(), tokens, True


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 64, 65536])
def test_chunked_corpus_prefix_equals_independent_line_oracle(tmp_path, chunk_size):
    rng = random.Random(2013)
    examples = [
        b"",
        b"a b  \r\n\n c d\n",
        b"a b " + b" " * 200000 + b"c\n",
        b"a" * 200000 + b" b\n",
        b"a b\n   ",
        b"\n\n\n",
    ]
    # Keep byte-at-a-time stress bounded; large cases exercise real chunk sizes.
    if chunk_size <= 7:
        examples = [example[:256] for example in examples]
    examples += [
        bytes(rng.choice(b"abc \t\r\n\v\f") for _ in range(200)) for _ in range(20)
    ]
    for data in examples:
        for budget in (0, 1, 2, 7, 1000):
            chunks = (data[i : i + chunk_size] for i in range(0, len(data), chunk_size))
            out = io.BytesIO()
            result = _copy_corpus_prefix(chunks, out, budget, scratch=tmp_path)
            expected, tokens, complete = reference(data, budget)
            assert (out.getvalue(), *result) == (expected, tokens, complete)


def test_materializer_long_line_and_whitespace_use_bounded_memory(tmp_path):
    peaks = []
    for count in (2000, 200000):
        source = tmp_path / "source"
        with source.open("wb") as out:
            for _ in range(count):
                out.write(b"word01234567890 ")
        tracemalloc.start()
        with source.open("rb") as stream, (tmp_path / "output").open("wb") as out:
            chunks = iter(lambda: stream.read(65536), b"")
            assert _copy_corpus_prefix(chunks, out, count, scratch=tmp_path) == (
                count,
                True,
            )
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    assert peaks[1] < peaks[0] * 2 + 1_000_000


def test_row_index_handles_many_queries_without_repeated_scans(tmp_path, monkeypatch):
    import gc
    import sqlite3

    from f2.suites.w2v import artifacts
    from repro_core.context import RuntimePaths

    monkeypatch.setattr(
        RuntimePaths,
        "from_environment",
        lambda: type("Paths", (), {"cache_root": tmp_path})(),
    )
    tokens = [f"word-{i}".encode() for i in range(10000)]
    packed = np.frombuffer(b"".join(tokens), dtype=np.uint8)
    offsets = np.array([0, *np.cumsum([len(t) for t in tokens])], dtype=np.uint64)
    lookup = artifacts.LookupEmbeddings(
        np.ones((len(tokens), 2), dtype=np.float32),
        packed,
        offsets,
        np.ones(len(tokens)),
        {},
    )
    assert lookup.row(tokens[-1]) == len(tokens) - 1
    directory = Path(lookup._index.execute("PRAGMA database_list").fetchone()[2]).parent

    # Each subsequent query must use a primary-key index, never token-array scans.
    class NoRead:
        def __getitem__(self, index):
            pytest.fail("lookup rescanned token bytes")

    lookup.token_bytes = NoRead()
    for i in range(0, 10000, 7):
        assert lookup.row(tokens[i]) == i
    assert lookup.row(b"missing") is None
    plan = lookup._index.execute(
        "EXPLAIN QUERY PLAN SELECT row FROM rows WHERE token=?", (tokens[-1],)
    ).fetchone()
    assert "PRIMARY KEY" in plan[-1]
    connection = lookup._index
    del lookup
    gc.collect()
    assert not directory.exists()
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
