"""Integration tests for PostgreSQL operational state repository, migrations, and Parquet export."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import psycopg
import pytest

from f2.corpus.analysis import FeasibilityAnalyzer
from f2.corpus.db.migrations.runner import run_migrations
from f2.corpus.db.repository import CorpusStateRepository
from f2.corpus.db.session import get_connection
from f2.corpus.discovery import CandidateRecord
from f2.corpus.pipeline import ProcessedDocumentResult
from f2.corpus.storage import ProvenanceExporter


@pytest.fixture
def db_conn():
    url = os.getenv("F2_CORPUS_DATABASE_URL")
    if not url:
        pytest.skip("F2_CORPUS_DATABASE_URL environment variable is not set")
    with get_connection(url) as conn:
        run_migrations(conn)
        yield conn


def test_migrations_and_idempotency(db_conn: psycopg.Connection):
    # Running migrations again should be a clean no-op
    applied = run_migrations(db_conn)
    assert applied == []


def test_repository_run_lifecycle_and_resumability(
    db_conn: psycopg.Connection, tmp_path: Path
):
    repo = CorpusStateRepository(db_conn)
    run_id = f"test_run_{uuid.uuid4().hex[:8]}"

    repo.create_run(
        run_id=run_id,
        run_type="test",
        crawl_ids=["CC-MAIN-2012"],
        sample_size=10,
        seed=42,
        bandwidth_mbps=20.0,
        concurrency=2,
        output_dir=tmp_path.as_posix(),
    )

    # Insert mock candidates
    cands = [
        CandidateRecord(
            crawl_id="CC-MAIN-2012",
            url=f"http://test{i}.com",
            timestamp="20120101",
            filename="cdx-00000.gz",
            offset=i * 100,
            length=100,
            digest=f"digest_{i}",
            source_type="probability_sample",
            stratum="unbiased",
            inclusion_probability=0.01,
            design_weight=100.0,
            block_index=0,
            record_index_in_block=i,
            block_total_records=10,
        )
        for i in range(5)
    ]
    inserted = repo.insert_candidates(run_id, cands)
    assert inserted == 5

    # Check initially completed candidate IDs
    completed = repo.get_completed_candidate_ids(run_id)
    assert len(completed) == 0

    # Record 2 successful results
    for i in range(2):
        c = cands[i]
        res = ProcessedDocumentResult(
            record_id=c.record_id(),
            crawl_id=c.crawl_id,
            url=c.url,
            fetch_status="success",
            downloaded_bytes=5000,
            http_status=200,
            extraction_success=True,
            clean_text="Sample text content for test document.",
            news_score=2.5,
            is_news_predicted=True,
            is_english=True,
            is_valid=True,
            rejection_reason=None,
            word_count=500,
            inclusion_probability=c.inclusion_probability,
            design_weight=c.design_weight,
            proxy_words=500,
            diagnostics={"detected_lang": "en"},
        )
        repo.record_processing_result(
            run_id, res, clean_text_sha256="abc", shard_path="shards/shard_00000.txt"
        )

    completed = repo.get_completed_candidate_ids(run_id)
    assert len(completed) == 2
    assert cands[0].record_id() in completed
    assert cands[1].record_id() in completed

    # Export to Parquet and JSONL
    exporter = ProvenanceExporter(repo)
    exports = exporter.export(run_id, tmp_path)
    assert exports["parquet"].exists()
    assert exports["jsonl"].exists()

    # Verify FeasibilityAnalyzer loads exported Parquet
    analyzer = FeasibilityAnalyzer(exports["parquet"])
    funnel = analyzer.compute_funnel_summary()
    assert funnel["total_sampled"] == 2
    assert funnel["valid_news"] == 2
    assert funnel["avg_words_per_doc"] == 500.0


def test_audit_workflow_in_repository(db_conn: psycopg.Connection):
    repo = CorpusStateRepository(db_conn)
    run_id = f"test_audit_run_{uuid.uuid4().hex[:8]}"

    repo.create_run(
        run_id=run_id,
        run_type="test_audit",
        crawl_ids=["CC-MAIN-2012"],
        sample_size=2,
        seed=42,
        bandwidth_mbps=20.0,
        concurrency=1,
        output_dir="/tmp",
    )

    c = CandidateRecord(
        crawl_id="CC-MAIN-2012",
        url="http://news.com/article",
        timestamp="20120101",
        filename="cdx-00000.gz",
        offset=0,
        length=100,
        digest="d1",
        source_type="prob",
        stratum="unbiased",
        inclusion_probability=0.01,
        design_weight=100.0,
        block_index=0,
        record_index_in_block=0,
        block_total_records=1,
    )
    repo.insert_candidates(run_id, [c])

    res = ProcessedDocumentResult(
        record_id=c.record_id(),
        crawl_id=c.crawl_id,
        url=c.url,
        fetch_status="success",
        downloaded_bytes=1000,
        http_status=200,
        extraction_success=True,
        clean_text="News article.",
        news_score=2.0,
        is_news_predicted=True,
        is_english=True,
        is_valid=True,
        rejection_reason=None,
        word_count=450,
        inclusion_probability=0.01,
        design_weight=100.0,
        proxy_words=450,
        diagnostics={},
    )
    repo.record_processing_result(run_id, res)

    # Insert audit assignment
    assignments = [
        {
            "record_id": c.record_id(),
            "predicted_class": 1,
            "priority_order": 0,
            "wave": 1,
            "audit_inclusion_prob_cond": 0.5,
            "audit_weight_cond": 2.0,
        }
    ]
    repo.insert_audit_assignments(run_id, assignments)

    # Record gold label with residual (gold = 500 words vs proxy = 450 words -> residual = +50)
    repo.record_audit_gold_label(
        run_id, c.record_id(), gold_class=1, word_count_gold=500, auditor_id="expert_1"
    )

    # Verify provenance includes audit gold annotations and residual
    prov = repo.get_provenance_records(run_id)
    assert len(prov) == 1
    assert prov[0]["is_audited"] is True
    assert prov[0]["word_count_gold"] == 500
    assert prov[0]["word_residual"] == 50
