from __future__ import annotations

import math
from pathlib import Path

import typer
from repro_io.commoncrawl.cdx import CDXIndexReader

from ...db.migrations import run_migrations
from ...db.repository import CorpusStateRepository
from ...db.session import get_connection
from ..discovery import TwoStageProbabilitySampler
from ..pipeline import PipelineRunner
from ..storage import CleanTextWriter, ProvenanceExporter


def execute_sampling(
    *,
    active_run_id: str,
    crawl_list: list[str],
    sample_size: int,
    per_crawl_target: int,
    seed: int,
    bandwidth_limit: float,
    concurrency: int,
    prefetch_rule: str,
    reject_exploration_rate: float,
    target_output_dir: Path,
    run_meta: dict[str, object],
) -> None:
    from f2_cc.corpus import cli

    with get_connection() as conn:
        run_migrations(conn)
        repo = CorpusStateRepository(conn)
        repo.create_run(
            run_id=active_run_id,
            run_type="sample",
            crawl_ids=crawl_list,
            sample_size=sample_size,
            seed=seed,
            bandwidth_mbps=bandwidth_limit,
            concurrency=concurrency,
            output_dir=target_output_dir.as_posix(),
            metadata=run_meta,
        )

        completed_candidate_ids = repo.get_completed_candidate_ids(active_run_id)
        if completed_candidate_ids:
            typer.echo(
                f"Resuming run: found {len(completed_candidate_ids)} already completed candidates in DB."
            )

        text_writer = CleanTextWriter(target_output_dir / "clean_shards")
        fetcher = cli.RangeFetcher(
            user_agent="abstract-dissection-repro/0.1 (Research reproduction study)",
            bandwidth_mbps=bandwidth_limit,
            max_concurrency=concurrency,
        )
        runner = PipelineRunner()

        typer.echo(
            f"Starting feasibility sampling across {len(crawl_list)} crawls (Target: {sample_size} records)..."
        )
        typer.echo(
            f"Prefetch Rule: {prefetch_rule}, Reject Rate: {reject_exploration_rate}, Concurrency: {concurrency}, Limit: {bandwidth_limit} Mbps"
        )

        records_per_block = 5
        blocks_per_crawl = max(1, math.ceil(per_crawl_target / records_per_block))

        processed_total = 0
        fetched_total = 0
        skipped_total = 0
        news_total = 0

        for crawl_idx, crawl_id in enumerate(crawl_list):
            typer.echo(
                f"\n--- [{crawl_idx + 1}/{len(crawl_list)}] Sampling Crawl: {crawl_id} ---"
            )
            reader = cli.ensure_cluster_index(crawl_id)
            sampler = TwoStageProbabilitySampler(
                crawl_id, reader, seed=seed + crawl_idx * 1000
            )

            blocks = sampler.plan_stage1_blocks(num_blocks=blocks_per_crawl)
            typer.echo(
                f"Selected {len(blocks)} primary CDX blocks out of {reader.total_blocks():,} total blocks."
            )

            for b_idx, block in enumerate(blocks):
                cdx_fetch = fetcher.fetch_cdx_block(crawl_id, block)
                if cdx_fetch.status_code not in {200, 206} or not cdx_fetch.data:
                    typer.echo(
                        f"  [Block {b_idx + 1}/{len(blocks)}] CDX fetch failed (HTTP {cdx_fetch.status_code}): {cdx_fetch.error_message}"
                    )
                    continue

                records = CDXIndexReader.parse_block_records(cdx_fetch.data)
                sampled_candidates = sampler.sample_block_records(
                    block, records, num_records_per_block=records_per_block
                )
                final_candidates = sampler.finalize_inclusion_probabilities(
                    sampled_candidates,
                    num_selected_blocks=len(blocks),
                    total_crawl_blocks=reader.total_blocks(),
                    prefetch_rule=prefetch_rule,
                    reject_exploration_rate=reject_exploration_rate,
                )

                repo.insert_candidates(active_run_id, final_candidates)

                for cand in final_candidates:
                    cand_id = cand.record_id()
                    if cand_id in completed_candidate_ids:
                        continue

                    if not cand.is_selected_for_fetch:
                        # Record skipped reject without issuing network request
                        skipped_res = runner.process(
                            record_id=cand_id,
                            crawl_id=cand.crawl_id,
                            url=cand.url,
                            raw_arc_compressed=b"",
                            inclusion_probability=cand.inclusion_probability,
                            design_weight=cand.design_weight,
                            downloaded_bytes=0,
                        )
                        repo.record_processing_result(
                            active_run_id,
                            skipped_res,
                            prefilter_status="reject",
                            is_reject_exploration=False,
                        )
                        completed_candidate_ids.add(cand_id)
                        skipped_total += 1
                        processed_total += 1
                        continue

                    # Fetch payload for pass stream or sampled reject exploration
                    arc_fetch = fetcher.fetch_range(
                        cand.filename, cand.offset, cand.length
                    )
                    is_rej_explore = cand.prefilter_status == "reject"
                    result = runner.process(
                        record_id=cand_id,
                        crawl_id=cand.crawl_id,
                        url=cand.url,
                        raw_arc_compressed=arc_fetch.data,
                        inclusion_probability=cand.inclusion_probability,
                        design_weight=cand.design_weight,
                        downloaded_bytes=arc_fetch.downloaded_bytes,
                    )

                    clean_sha = None
                    shard_p = None
                    if (
                        result.is_valid
                        and result.is_news_predicted
                        and result.is_english
                        and result.clean_text
                    ):
                        clean_sha, shard_p = text_writer.write_document(
                            result.clean_text, result.word_count, cand.url
                        )
                        news_total += 1

                    repo.record_processing_result(
                        active_run_id,
                        result,
                        clean_text_sha256=clean_sha,
                        shard_path=shard_p,
                        prefilter_status=cand.prefilter_status,
                        is_reject_exploration=is_rej_explore,
                    )
                    completed_candidate_ids.add(cand_id)
                    fetched_total += 1
                    processed_total += 1

                    if processed_total % 250 == 0 or processed_total == sample_size:
                        typer.echo(
                            f"  [Progress: {processed_total}/{sample_size}] Fetched: {fetched_total}, Skipped Rejects: {skipped_total}, Retained News: {news_total}"
                        )

        text_writer.close()
        repo.update_run_status(active_run_id, "completed")

        # Export provenance to parquet & jsonl
        exporter = ProvenanceExporter(repo)
        exports = exporter.export(active_run_id, target_output_dir)
        typer.echo(f"\nSampling completed successfully! Run ID: {active_run_id}")
        typer.echo(f"  - Total Candidates: {processed_total}")
        typer.echo(f"  - Fetched Payloads: {fetched_total}")
        typer.echo(
            f"  - Avoided Rejects:  {skipped_total} ({skipped_total / max(1, processed_total) * 100:.1f}%)"
        )
        typer.echo(f"  - Retained News:    {news_total}")
        typer.echo(f"  - Parquet Export:   {exports['parquet']}")
        typer.echo(f"  - JSONL Export:     {exports['jsonl']}")
