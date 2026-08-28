"""Regression and end-to-end CLI tests ensuring HTTP 206 range responses and non-empty provenance exports."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import psycopg
import pytest
from typer.testing import CliRunner

from f2.cli import app, ensure_cluster_index
from f2.corpus.db.migrations.runner import run_migrations
from f2.corpus.db.session import get_connection
from f2.fetcher import FetchResult

runner = CliRunner()


@pytest.fixture
def db_ready():
    url = os.getenv("F2_CORPUS_DATABASE_URL")
    if not url:
        pytest.skip("F2_CORPUS_DATABASE_URL environment variable is not set")
    with get_connection(url) as conn:
        run_migrations(conn)
        yield conn


def test_cli_sample_accepts_http_206_and_produces_non_empty_provenance(
    db_ready: psycopg.Connection, tmp_path: Path
):
    """Regression test: Ensure HTTP 206 range response is accepted and yields non-empty provenance."""
    # Construct valid mock CDX block (3 records)
    cdx_lines = [
        'com,reuters)/article1 20120101 {"status":"200","url":"http://reuters.com/1","filename":"seg1.arc.gz","offset":"0","length":"500","mime":"text/html"}',
        'com,reuters)/article2 20120101 {"status":"200","url":"http://reuters.com/2","filename":"seg1.arc.gz","offset":"500","length":"500","mime":"text/html"}',
        'com,reuters)/article3 20120101 {"status":"200","url":"http://reuters.com/3","filename":"seg1.arc.gz","offset":"1000","length":"500","mime":"text/html"}',
    ]
    cdx_block_bytes = gzip.compress("\n".join(cdx_lines).encode("utf-8"))

    # Construct valid mock ARC record (news article)
    html_content = (
        "<html><body><article>"
        "<h1>Global Economic Recovery Accelerates in Q1</h1>"
        "<p>By Jane Doe, Reuters</p><p>March 15, 2012</p>"
        "<p>LONDON — Global economic activity expanded at its fastest pace in two years during the first quarter, "
        "driven by robust industrial output and rising consumer confidence across major economies.</p>"
        '<p>"We are seeing broad-based momentum across key manufacturing sectors," said John Smith, senior economist at Global Bank. '
        "He stated that corporate investments have rebounded significantly following policy easing.</p>"
        "<p>Financial markets rallied following the release, with equity benchmarks gaining over two percent. "
        "Analysts reported that sovereign bond yields remained steady, indicating stable inflation expectations worldwide.</p>"
        "</article></body></html>"
    )
    http_payload = f"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {len(html_content)}\r\n\r\n{html_content}".encode()
    arc_header = f"http://reuters.com/1 127.0.0.1 20120101000000 text/html {len(http_payload)}\n".encode()
    arc_bytes = gzip.compress(arc_header + http_payload)

    # Mock RangeFetcher to return HTTP 206 Partial Content (the exact Common Crawl behavior)
    mock_fetcher = MagicMock()
    mock_fetcher.fetch_cdx_block.return_value = FetchResult(
        status_code=206,
        data=cdx_block_bytes,
        downloaded_bytes=len(cdx_block_bytes),
        elapsed_sec=0.01,
    )
    mock_fetcher.fetch_range.return_value = FetchResult(
        status_code=206,
        data=arc_bytes,
        downloaded_bytes=len(arc_bytes),
        elapsed_sec=0.01,
    )

    out_dir = tmp_path / "sample_out"
    with patch("f2.cli.RangeFetcher", return_value=mock_fetcher):
        result = runner.invoke(
            app,
            [
                "sample",
                "--crawls",
                "CC-MAIN-2012",
                "--sample-size",
                "2",
                "--seed",
                "42",
                "--output-dir",
                str(out_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Sampling completed successfully!" in result.output

    # Crucial regression assertions: files must exist AND have > 0 records
    parquet_file = out_dir / "provenance.parquet"
    jsonl_file = out_dir / "provenance.jsonl"
    assert parquet_file.exists()
    assert jsonl_file.exists()

    jsonl_lines = [
        json.loads(line)
        for line in jsonl_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(jsonl_lines) > 0, (
        "Regression detected: provenance.jsonl was empty (0 records processed)"
    )

    # Verify document extraction and news classification
    first_record = jsonl_lines[0]
    assert first_record["fetch_status"] == "success"
    assert first_record["extraction_success"] == 1
    assert first_record["is_news_predicted"] == 1
    assert first_record["word_count"] > 50


def test_ensure_cluster_index_cached(tmp_path: Path):
    """Test ensure_cluster_index returns indexed reader from existing cache without network request."""
    sample_cluster_idx = (
        "com,apple)/ 20120101000000\tcdx-00000.gz\t0\t200000\t1\n"
        "com,bbc)/ 20120101000000\tcdx-00000.gz\t200000\t200000\t2\n"
    )
    # Pad to > 10,000 bytes to simulate real index file
    padded = sample_cluster_idx + ("# comment\n" * 1500)
    cache_file = tmp_path / "cluster.idx"
    cache_file.write_text(padded, encoding="utf-8")

    with patch("repro_core.context.paths.RuntimePaths.from_environment") as mock_paths:
        mock_paths.return_value.cache_root = tmp_path.parent
        # Create expected directory structure
        crawl_dir = tmp_path.parent / "f2" / "CC-MAIN-2012"
        crawl_dir.mkdir(parents=True, exist_ok=True)
        (crawl_dir / "cluster.idx").write_text(padded, encoding="utf-8")

        reader = ensure_cluster_index("CC-MAIN-2012")
        assert reader.total_blocks() == 2
