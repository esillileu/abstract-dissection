"""Live network smoke test against real Common Crawl 2012 and 2009-2010 endpoints with PostgreSQL state."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from f2.analysis import FeasibilityAnalyzer
from f2.cdx import CDXBlockLocator, CDXIndexReader
from f2.corpus.db.migrations.runner import run_migrations
from f2.corpus.db.repository import CorpusStateRepository
from f2.corpus.db.session import get_connection
from f2.discovery import TwoStageProbabilitySampler
from f2.fetcher import RangeFetcher
from f2.pipeline import ARCParser, PipelineRunner
from f2.storage import CleanTextWriter, ProvenanceExporter


@pytest.mark.network
def test_live_common_crawl_2012_range_fetch():
    fetcher = RangeFetcher(bandwidth_mbps=20.0, max_concurrency=1)
    # Known valid ARC slice from CC-MAIN-2012
    filename = "parse-output/segment/1346876860565/1346911687874_322.arc.gz"
    offset = 29913880
    length = 792

    res = fetcher.fetch_range(filename, offset, length)
    assert res.status_code in {200, 206}
    assert res.downloaded_bytes == length
    assert len(res.data) == length

    arc_record = ARCParser.parse_arc_bytes(res.data)
    assert arc_record is not None
    assert arc_record.http_status == 200


@pytest.mark.network
def test_live_common_crawl_2009_2010_range_fetch():
    fetcher = RangeFetcher(bandwidth_mbps=20.0, max_concurrency=1)
    # Known valid ARC slice from CC-MAIN-2009-2010
    filename = "crawl-002/2009/09/17/26/1253232078673_26.arc.gz"
    offset = 67118456
    length = 13087

    res = fetcher.fetch_range(filename, offset, length)
    assert res.status_code in {200, 206}
    assert res.downloaded_bytes == length

    arc_record = ARCParser.parse_arc_bytes(res.data)
    assert arc_record is not None
    assert arc_record.http_status == 200


@pytest.mark.network
def test_live_end_to_end_smoke_with_postgres(tmp_path: Path):
    db_url = os.getenv("F2_CORPUS_DATABASE_URL")
    if not db_url:
        pytest.skip("F2_CORPUS_DATABASE_URL is not set")

    run_id = f"smoke_{uuid.uuid4().hex[:8]}"
    fetcher = RangeFetcher(bandwidth_mbps=20.0, max_concurrency=2)
    runner = PipelineRunner(min_words=50)
    text_writer = CleanTextWriter(tmp_path / "shards")

    with get_connection(db_url) as conn:
        run_migrations(conn)
        repo = CorpusStateRepository(conn)
        repo.create_run(
            run_id=run_id,
            run_type="smoke",
            crawl_ids=["CC-MAIN-2009-2010", "CC-MAIN-2012"],
            sample_size=4,
            seed=42,
            bandwidth_mbps=20.0,
            concurrency=2,
            output_dir=tmp_path.as_posix(),
        )

        # 1. Fetch 1 real CDX block from CC-MAIN-2012
        block_2012 = CDXBlockLocator(
            surt_key="a)",
            timestamp="20120101",
            filename="cdx-00000.gz",
            offset=0,
            length=208582,
            block_index=0,
        )
        cdx_res_2012 = fetcher.fetch_cdx_block("CC-MAIN-2012", block_2012)
        assert cdx_res_2012.status_code in {200, 206}
        records_2012 = CDXIndexReader.parse_block_records(cdx_res_2012.data)
        assert len(records_2012) == 3000

        # Sample 2 records from 2012 block
        reader_2012 = CDXIndexReader([block_2012])
        sampler_2012 = TwoStageProbabilitySampler("CC-MAIN-2012", reader_2012, seed=42)
        cands_2012 = sampler_2012.sample_block_records(
            block_2012, records_2012, num_records_per_block=2
        )
        final_cands_2012 = sampler_2012.finalize_inclusion_probabilities(
            cands_2012, 1, 1
        )

        # 2. Fetch 1 real CDX block from CC-MAIN-2009-2010
        block_2009 = CDXBlockLocator(
            surt_key="a)",
            timestamp="20090917",
            filename="cdx-00000.gz",
            offset=0,
            length=345003,
            block_index=0,
        )
        cdx_res_2009 = fetcher.fetch_cdx_block("CC-MAIN-2009-2010", block_2009)
        assert cdx_res_2009.status_code in {200, 206}
        records_2009 = CDXIndexReader.parse_block_records(cdx_res_2009.data)
        assert len(records_2009) == 3000

        # Sample 2 records from 2009 block
        reader_2009 = CDXIndexReader([block_2009])
        sampler_2009 = TwoStageProbabilitySampler(
            "CC-MAIN-2009-2010", reader_2009, seed=42
        )
        cands_2009 = sampler_2009.sample_block_records(
            block_2009, records_2009, num_records_per_block=2
        )
        final_cands_2009 = sampler_2009.finalize_inclusion_probabilities(
            cands_2009, 1, 1
        )

        all_cands = final_cands_2012 + final_cands_2009
        repo.insert_candidates(run_id, all_cands)

        # 3. Process candidate records against live Common Crawl ARC files
        for cand in all_cands:
            arc_res = fetcher.fetch_range(cand.filename, cand.offset, cand.length)
            assert arc_res.status_code in {200, 206}
            assert arc_res.downloaded_bytes == cand.length

            result = runner.process(
                record_id=cand.record_id(),
                crawl_id=cand.crawl_id,
                url=cand.url,
                raw_arc_compressed=arc_res.data,
                inclusion_probability=cand.inclusion_probability,
                design_weight=cand.design_weight,
                downloaded_bytes=arc_res.downloaded_bytes,
            )

            sha256_hash = None
            shard_path = None
            if result.clean_text:
                sha256_hash, shard_path = text_writer.write_document(
                    result.clean_text, result.word_count, cand.url
                )

            repo.record_processing_result(
                run_id, result, clean_text_sha256=sha256_hash, shard_path=shard_path
            )

        text_writer.close()
        repo.update_run_status(run_id, "completed")

        # 4. Verify Resumability / Idempotence
        completed_ids = repo.get_completed_candidate_ids(run_id)
        assert len(completed_ids) == len(all_cands)

        # 5. Export to Parquet and JSONL
        exporter = ProvenanceExporter(repo)
        exports = exporter.export(run_id, tmp_path)
        assert exports["parquet"].exists()
        assert exports["jsonl"].exists()

        # 6. Verify DuckDB Feasibility Analysis over exported Parquet
        analyzer = FeasibilityAnalyzer(exports["parquet"])
        funnel = analyzer.compute_funnel_summary()
        assert funnel["total_sampled"] == 4
        assert funnel["fetch_success"] == 4
        assert funnel["avg_download_bytes"] > 0
