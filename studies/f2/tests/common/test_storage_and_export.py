"""Tests for CleanTextWriter and storage helpers in f2.common.storage."""

from __future__ import annotations

from pathlib import Path

from f2.common.storage import CleanTextWriter


def test_clean_text_writer(tmp_path: Path):
    out_dir = tmp_path / "clean_shards"
    writer = CleanTextWriter(out_dir, max_words_per_shard=100)

    sha1, _ = writer.write_document("Hello world news article.", 4, "http://test.com/1")
    sha2, _ = writer.write_document("Second article body text.", 4, "http://test.com/2")
    writer.close()

    assert sha1 != ""
    assert sha2 != ""
    shards = list(out_dir.glob("shard_*.txt"))
    assert len(shards) == 1
    content = shards[0].read_text(encoding="utf-8")
    assert '<DOC url="http://test.com/1" words="4">' in content
    assert "Hello world news article." in content
