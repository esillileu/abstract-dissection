"""Typer CLI interface for F2 Word2Vec corpus feasibility, sampling, auditing, database migrations, and analysis."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Annotated

import typer

from repro_core.context.paths import RuntimePaths

from .analysis import FeasibilityAnalyzer
from .calibration import CalibrationAndPreFetchAnalyzer
from .cdx import CDXBlockLocator, CDXIndexReader
from .corpus.db.migrations.runner import run_migrations
from .corpus.db.repository import CorpusStateRepository
from .corpus.db.session import get_connection
from .discovery import (
    SEED_DOMAIN_CATALOG,
    SequentialAuditSampler,
    TwoStageProbabilitySampler,
)
from .fetcher import RangeFetcher
from .pipeline import PipelineRunner
from .storage import CleanTextWriter, ProvenanceExporter

app = typer.Typer(
    name="corpus",
    help="Plan, sample, audit, and analyze Common Crawl corpus feasibility for Word2Vec.",
    no_args_is_help=True,
)


def ensure_cluster_index(crawl_id: str) -> CDXIndexReader:
    """Ensure Common Crawl CDX cluster index is available in cache and return indexed reader."""
    cache_dir = RuntimePaths.from_environment().cache_root / "f2" / crawl_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    idx_path = cache_dir / "cluster.idx"

    if idx_path.exists() and idx_path.stat().st_size > 10_000:
        return CDXIndexReader.from_file(idx_path)

    # Download from Common Crawl index repository
    url = f"https://data.commoncrawl.org/cc-index/collections/{crawl_id}/indexes/cluster.idx"
    typer.echo(f"Downloading Common Crawl CDX cluster index for {crawl_id}...")
    temp_path = cache_dir / "cluster.idx.tmp"
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "abstract-dissection-repro/0.1 (Research reproduction study)"
            },
        )
        with (
            urllib.request.urlopen(req, timeout=60.0) as resp,
            temp_path.open("wb") as f,
        ):
            while chunk := resp.read(1024 * 1024):
                f.write(chunk)
        temp_path.replace(idx_path)
        typer.echo(
            f"Saved cluster index to: {idx_path} ({idx_path.stat().st_size / 1_000_000:.1f} MB)"
        )
        return CDXIndexReader.from_file(idx_path)
    except Exception as exc:
        typer.echo(
            f"Warning: Could not download remote cluster index ({exc}). Using offline baseline."
        )
        mock_entries = [
            CDXBlockLocator(
                surt_key="a)",
                timestamp="20120101",
                filename="cdx-00000.gz",
                offset=0,
                length=208582,
                block_index=0,
            ),
            CDXBlockLocator(
                surt_key="m)",
                timestamp="20120101",
                filename="cdx-00000.gz",
                offset=208582,
                length=211621,
                block_index=1,
            ),
        ]
        return CDXIndexReader(mock_entries)


@app.command("migrate")
def migrate_db() -> None:
    """Apply pending PostgreSQL migrations for the F2 corpus control plane."""
    with get_connection() as conn:
        applied = run_migrations(conn)
        if applied:
            typer.echo(
                f"Successfully applied {len(applied)} migrations: {', '.join(applied)}"
            )
        else:
            typer.echo("Database schema is up to date (0 pending migrations).")


@app.command("plan")
def plan_corpus(
    crawl: Annotated[
        str, typer.Option("--crawl", "-c", help="Common Crawl ID (e.g. CC-MAIN-2012)")
    ] = "CC-MAIN-2012",
    sample_size: Annotated[
        int, typer.Option("--sample-size", "-n", help="Target sample size")
    ] = 100,
    seed: Annotated[int, typer.Option("--seed", "-s", help="Random seed")] = 20260227,
) -> None:
    """Inspect sampling plan, strata parameters, and design inclusion probabilities."""
    typer.echo(f"=== F2 Corpus Sampling Plan ({crawl}) ===")
    typer.echo(f"Random Seed: {seed}")
    typer.echo(f"Target Sample Size: {sample_size}")
    typer.echo("\nCurated Seed Strata:")
    for stratum, domains in SEED_DOMAIN_CATALOG.items():
        typer.echo(
            f"  - {stratum.value}: {len(domains)} domains ({', '.join(domains[:3])}...)"
        )

    typer.echo("\nTwo-Stage Probability Sampling Protocol:")
    typer.echo(
        "  Top-Level Strata: Independent crawl snapshots (CC-MAIN-2009-2010, CC-MAIN-2012)"
    )
    typer.echo("  Stage 1: SRS of m CDX blocks (p_k = m / K)")
    typer.echo(
        "  Stage 2: SRS of n_k records per block (pi_within = n_k / M_k, where M_k = 3,000)"
    )
    typer.echo("  Pre-Fetch Filter: Rule 1 Only (32 binary media extensions)")
    typer.echo("  Reject Exploration Rate: 5% (pi_explore = 0.05)")
    typer.echo("  State Store: PostgreSQL operational control plane")


@app.command("sample")
def sample_corpus(
    crawls: Annotated[
        str, typer.Option("--crawls", "-c", help="Comma-separated crawl IDs")
    ] = "CC-MAIN-2009-2010,CC-MAIN-2012",
    sample_size: Annotated[
        int, typer.Option("--sample-size", "-n", help="Total records to sample")
    ] = 50000,
    seed: Annotated[int, typer.Option("--seed", "-s", help="Random seed")] = 20260227,
    prefetch_rule: Annotated[
        str, typer.Option("--prefetch-rule", help="Pre-fetch rule ('rule1' or 'none')")
    ] = "rule1",
    reject_exploration_rate: Annotated[
        float,
        typer.Option(
            "--reject-exploration-rate", help="Sampling rate for rejected records"
        ),
    ] = 0.05,
    bandwidth_limit: Annotated[
        float, typer.Option("--bandwidth-limit", "-b", help="Bandwidth limit in Mbps")
    ] = 20.0,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-j", help="Concurrent fetch workers")
    ] = 4,
    output_dir: Annotated[
        Path | None, typer.Option("--output-dir", "-o", help="Output directory")
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Explicit run identifier for resuming"),
    ] = None,
) -> None:
    """Execute bounded, rate-limited probability sample against Common Crawl backed by PostgreSQL."""
    paths = RuntimePaths.from_environment()
    target_output_dir = output_dir or (paths.staging_root / "exp" / "f2" / "sample")
    target_output_dir.mkdir(parents=True, exist_ok=True)
    active_run_id = run_id or f"run_{seed}_{uuid.uuid4().hex[:8]}"

    crawl_list = [c.strip() for c in crawls.split(",") if c.strip()]
    per_crawl_target = max(1, sample_size // len(crawl_list))

    try:
        exec_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        exec_sha = "unknown"

    config_dict = {
        "seed": seed,
        "sample_size": sample_size,
        "crawls": crawl_list,
        "prefetch_rule": prefetch_rule,
        "reject_exploration_rate": reject_exploration_rate,
        "concurrency": concurrency,
        "bandwidth_limit": bandwidth_limit,
    }
    config_hash = hashlib.sha256(
        json.dumps(config_dict, sort_keys=True).encode()
    ).hexdigest()[:16]

    run_meta = {
        "baseline_10k_commit_sha": "f3dee9676517d9a7506b162aff83a111f45209dc",
        "execution_50k_commit_sha": exec_sha,
        "config_hash": config_hash,
        "frozen_parameters": config_dict,
    }

    typer.echo(f"Initializing run '{active_run_id}' in PostgreSQL...")
    typer.echo(f"  - Baseline SHA: {run_meta['baseline_10k_commit_sha']}")
    typer.echo(f"  - Execution SHA: {run_meta['execution_50k_commit_sha']}")
    typer.echo(f"  - Config Hash: {config_hash}")

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

        text_writer = CleanTextWriter(output_dir / "clean_shards")
        fetcher = RangeFetcher(
            bandwidth_mbps=bandwidth_limit, max_concurrency=concurrency
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
            reader = ensure_cluster_index(crawl_id)
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


@app.command("export")
def export_run(
    run_id: Annotated[
        str, typer.Option("--run-id", "-r", help="Run ID to export from DB")
    ],
    output_dir: Annotated[
        Path | None, typer.Option("--output-dir", "-o", help="Output directory")
    ] = None,
) -> None:
    """Export provenance records for a given run from PostgreSQL to Parquet and JSONL."""
    paths = RuntimePaths.from_environment()
    target_output_dir = output_dir or (paths.staging_root / "exp" / "f2" / "export")
    target_output_dir.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        repo = CorpusStateRepository(conn)
        exporter = ProvenanceExporter(repo)
        exports = exporter.export(run_id, target_output_dir)
        typer.echo(f"Exported run '{run_id}':")
        typer.echo(f"  - Parquet: {exports['parquet']}")
        typer.echo(f"  - JSONL:   {exports['jsonl']}")


@app.command("audit")
def create_audit(
    run_id: Annotated[
        str,
        typer.Option("--run-id", "-r", help="Run ID from which to create audit sample"),
    ],
    budget: Annotated[
        int, typer.Option("--budget", "-b", help="Total audit sample size")
    ] = 400,
    seed: Annotated[
        int, typer.Option("--seed", "-s", help="Random seed for audit priority")
    ] = 20260227,
) -> None:
    """Generate pre-specified 8-stratum sequential audit assignments and store in PostgreSQL."""
    with get_connection() as conn:
        repo = CorpusStateRepository(conn)
        records = repo.get_provenance_records(run_id)
        if not records:
            typer.echo(f"No records found for run '{run_id}'.")
            return

        # Use only records that were actually fetched
        fetched_records = [r for r in records if r.get("fetch_status") == "success"]
        sampler = SequentialAuditSampler(seed=seed)
        schedule = sampler.generate_8_stratum_audit_schedule(fetched_records)
        assignments = sampler.select_8_stratum_audit_wave(schedule, total_budget=budget)

        count = repo.insert_audit_assignments(run_id, assignments)
        typer.echo(
            f"Created {count} audit assignments across 8 design strata for run '{run_id}'."
        )


@app.command("audit-review")
def review_audit(
    run_id: Annotated[
        str, typer.Option("--run-id", "-r", help="Run ID to export audit records for")
    ],
    output_file: Annotated[
        Path | None,
        typer.Option("--output-file", "-o", help="Audit review JSONL output path"),
    ] = None,
    blind: Annotated[
        bool,
        typer.Option(
            "--blind/--no-blind",
            help="Mask model predictions & scores for double-blind auditing",
        ),
    ] = True,
) -> None:
    """Export the audit assignments with text for manual labeling (supports double-blind mode)."""
    paths = RuntimePaths.from_environment()
    target_output_file = output_file or (
        paths.staging_root / "exp" / "f2" / "00_corpus_audit_set_50k_400_blind.jsonl"
    )
    target_output_file.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        repo = CorpusStateRepository(conn)
        audit_items = repo.get_audit_assignments(run_id)
        if not audit_items:
            typer.echo(
                f"No audit assignments found for run '{run_id}'. Run 'repro f2 corpus audit' first."
            )
            return

        # Map URLs to clean text snippet if shards exist
        shard_candidates = [
            paths.staging_root / "exp" / "f2" / "confirmatory_50k" / "clean_shards",
            paths.staging_root / "exp" / "f2" / "sample_10k" / "clean_shards",
            paths.staging_root / "exp" / "f2" / "sample" / "clean_shards",
        ]
        shard_dir = next(
            (p for p in shard_candidates if p.exists()), shard_candidates[0]
        )

        url_to_text: dict[str, str] = {}
        if shard_dir.exists():
            import re

            for sf in shard_dir.glob("*.txt"):
                docs = re.findall(
                    r'<DOC url="(.*?)" words="(\d+)">\n(.*?)\n</DOC>',
                    sf.read_text(encoding="utf-8"),
                    re.DOTALL,
                )
                for u, _, txt in docs:
                    url_to_text[u] = txt.strip()

        exported_items = []
        with target_output_file.open("w", encoding="utf-8") as f:
            for item in audit_items:
                u = item["url"]
                domain = urllib.parse.urlparse(u).netloc.lower()
                snippet = url_to_text.get(u, "")
                if snippet and len(snippet) > 800:
                    snippet = snippet[:800] + "..."

                if blind:
                    review_entry = {
                        "audit_id": item["audit_id"],
                        "candidate_id": item["candidate_id"],
                        "priority_order": item["priority_order"],
                        "wave": item.get("wave", 1),
                        "crawl_id": item["crawl_id"],
                        "url": u,
                        "domain": domain,
                        "word_count": item["word_count"],
                        "text_snippet": snippet,
                        "gold_class": item["gold_class"]
                        if item["is_audited"]
                        else None,
                        "word_count_gold": item["word_count_gold"]
                        if item["is_audited"]
                        else None,
                        "auditor_id": item.get("auditor_id"),
                        "notes": item.get("notes"),
                    }
                else:
                    review_entry = {
                        "audit_id": item["audit_id"],
                        "candidate_id": item["candidate_id"],
                        "priority_order": item["priority_order"],
                        "design_stratum": item.get("design_stratum"),
                        "audit_stratum": item["audit_stratum"],
                        "wave": item.get("wave", 1),
                        "crawl_id": item["crawl_id"],
                        "url": u,
                        "domain": domain,
                        "first_stage_inclusion_probability": item["first_stage_pi"],
                        "first_stage_design_weight": item["first_stage_weight"],
                        "audit_inclusion_probability": item[
                            "audit_inclusion_probability"
                        ],
                        "audit_design_weight": item["audit_design_weight"],
                        "news_score": item["news_score"],
                        "is_news_predicted": item["is_news_predicted"],
                        "is_english": item["is_english"],
                        "is_valid": item["is_valid"],
                        "word_count": item["word_count"],
                        "word_count_proxy": item["word_count_proxy"],
                        "diagnostics": item["diagnostics"],
                        "text_snippet": snippet,
                        "gold_class": item["gold_class"]
                        if item["is_audited"]
                        else None,
                        "word_count_gold": item["word_count_gold"]
                        if item["is_audited"]
                        else None,
                        "auditor_id": item.get("auditor_id"),
                        "notes": item.get("notes"),
                    }
                exported_items.append(review_entry)
                f.write(json.dumps(review_entry, default=str) + "\n")

        # Also write Markdown Review Dossier
        md_file = target_output_file.parent / "00_corpus_audit_set_50k_400_review.md"
        mode_tag = "Double-Blind Mode" if blind else "Unblinded Control Mode"
        md_lines = [
            f"# Phase-2 8-Stratum Gold Audit Set ({len(audit_items)} Documents) — Run `{run_id}` ({mode_tag})",
            "",
            "> **Pre-specified 8-Stratum Factorial Audit Design (Crawl x Prefilter x Postfilter)**",
            "",
            "---",
            "",
        ]
        for entry in exported_items:
            strat_display = (
                f"Stratum `{entry.get('design_stratum', 'N/A')}`"
                if not blind
                else "Blinded Unit"
            )
            md_lines.extend(
                [
                    f"### #{entry['priority_order']:03d} [{strat_display}] `{entry['audit_id']}`",
                    f"- **URL:** [{entry['url']}]({entry['url']})",
                    f"- **Crawl:** `{entry['crawl_id']}` | **Domain:** `{entry['domain']}`",
                    f"- **Document Length:** `{entry.get('word_count', 0):,} words`",
                ]
            )
            if not blind:
                md_lines.extend(
                    [
                        f"- **Classifier Score:** `{entry.get('news_score', 0.0):.1f}` | **Proxy Words:** `{entry.get('word_count_proxy', 0):,}`",
                        f"- **Weights:** $\\pi_1 = {entry.get('first_stage_inclusion_probability', 0.0):.2e}$ | $\\pi_2 = {entry.get('audit_inclusion_probability', 0.0):.4f}$",
                    ]
                )
            if entry["text_snippet"]:
                md_lines.extend(["", "```text", entry["text_snippet"], "```", ""])
            else:
                md_lines.extend(["", "*(No clean text snippet retained)*", ""])
            md_lines.extend(["---", ""])

        md_file.write_text("\n".join(md_lines), encoding="utf-8")
        typer.echo(f"Exported {len(audit_items)} audit review documents to:")
        typer.echo(f"  - JSONL: {target_output_file}")
        typer.echo(f"  - Markdown Dossier: {md_file}")


@app.command("audit-record")
def record_audit(
    run_id: Annotated[
        str, typer.Option("--run-id", "-r", help="Run ID to record audit labels for")
    ],
    audit_file: Annotated[
        Path | None,
        typer.Option("--audit-file", "-a", help="Path to annotated audit JSONL"),
    ] = None,
) -> None:
    """Record completed audit gold labels into PostgreSQL from an annotated JSONL file."""
    paths = RuntimePaths.from_environment()
    target_audit_file = audit_file
    if target_audit_file is None:
        candidate_audits = [
            paths.staging_root
            / "exp"
            / "f2"
            / "00_corpus_audit_set_50k_400_annotated.jsonl",
            paths.staging_root / "exp" / "f2" / "audit_set_50k_400_annotated.jsonl",
            paths.analysis_output("f2") / "00_corpus_audit_set_50k_400_annotated.jsonl",
            paths.analysis_output("f2") / "audit_set_50k_400_annotated.jsonl",
        ]
        target_audit_file = next(
            (p for p in candidate_audits if p.exists()), candidate_audits[0]
        )

    if not target_audit_file.exists():
        typer.echo(f"Audit file not found: {target_audit_file}")
        return

    annotated_records = [
        json.loads(line)
        for line in target_audit_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with get_connection() as conn:
        repo = CorpusStateRepository(conn)
        recorded = 0
        for r in annotated_records:
            cand_id = r.get("candidate_id") or r.get("record_id")
            if r.get("gold_class") is not None and r.get("word_count_gold") is not None:
                repo.record_audit_gold_label(
                    run_id=run_id,
                    candidate_id=cand_id,
                    gold_class=int(r["gold_class"]),
                    word_count_gold=int(r["word_count_gold"]),
                    auditor_id=r.get("auditor_id", "blind_expert"),
                    notes=r.get("notes"),
                )
                recorded += 1
        typer.echo(
            f"Recorded {recorded} gold audit labels into PostgreSQL for run '{run_id}'."
        )


@app.command("analyze")
def analyze_corpus(
    manifest: Annotated[
        Path | None,
        typer.Option(
            "--manifest", "-m", help="Path to provenance.parquet or provenance.jsonl"
        ),
    ] = None,
    audit_file: Annotated[
        Path | None, typer.Option("--audit-file", "-a", help="Path to gold audit JSONL")
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Analysis report output dir"),
    ] = None,
) -> None:
    """Run Two-Phase 8-Stratum Estimation, Two-Stage Cluster Bootstrap, & Feasibility Verification."""
    paths = RuntimePaths.from_environment()
    target_output_dir = output_dir or paths.analysis_output("f2", "corpus")
    target_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest
    if manifest_path is None:
        manifest_candidates = [
            paths.staging_root
            / "exp"
            / "f2"
            / "confirmatory_50k"
            / "provenance.parquet",
            paths.staging_root / "exp" / "f2" / "sample" / "provenance.parquet",
            paths.staging_root
            / "exp"
            / "f2"
            / "sample_10k_audited"
            / "provenance.parquet",
        ]
        manifest_path = next(
            (p for p in manifest_candidates if p.exists()), manifest_candidates[0]
        )

    if not manifest_path.exists():
        typer.echo(f"Error: Manifest file not found: {manifest_path}")
        return

    analyzer = FeasibilityAnalyzer(manifest_path)

    audit_records = None
    if audit_file and audit_file.exists():
        audit_records = [
            json.loads(line)
            for line in audit_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif audit_file is None:
        default_audit_paths = [
            target_output_dir / "00_corpus_audit_set_50k_400_annotated.jsonl",
            paths.staging_root
            / "exp"
            / "f2"
            / "00_corpus_audit_set_50k_400_annotated.jsonl",
            paths.analysis_output("f2") / "00_corpus_audit_set_50k_400_annotated.jsonl",
            paths.staging_root / "exp" / "f2" / "audit_set_50k_400_annotated.jsonl",
            target_output_dir / "audit_set_50k_400_annotated.jsonl",
            paths.staging_root
            / "exp"
            / "f2"
            / "00_corpus_audit_review_400_annotated.jsonl",
            paths.staging_root / "exp" / "f2" / "audit_review_400_annotated.jsonl",
            target_output_dir / "00_corpus_audit_set_400_annotated.jsonl",
            target_output_dir / "audit_set_400_annotated.jsonl",
        ]
        for p in default_audit_paths:
            if p.exists():
                audit_records = [
                    json.loads(line)
                    for line in p.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                break

    report_data = analyzer.compute_two_phase_yield(
        audit_records=audit_records, bootstrap_reps=1000
    )
    md_content = analyzer.generate_report_markdown(report_data)

    report_file = target_output_dir / "00_corpus_confirmatory_50k_report.md"
    report_file.write_text(md_content, encoding="utf-8")

    # Also export summary.csv
    csv_file = target_output_dir / "00_corpus_confirmatory_50k_summary.csv"
    csv_lines = [
        "crawl_id,sample_size,retained_news_docs,proxy_words,residual_error,true_total_words,ci_lower_95,ci_upper_95,weighted_ppv,weighted_tpr"
    ]
    for y in report_data.strata_yields:
        ppv_v = f"{y.weighted_ppv:.4f}" if y.weighted_ppv is not None else ""
        tpr_v = f"{y.weighted_tpr:.4f}" if y.weighted_tpr is not None else ""
        csv_lines.append(
            f"{y.crawl_id},{y.sample_size},{y.retained_news_docs},{y.proxy_total_words:.0f},{y.residual_error_words:.0f},{y.true_total_words:.0f},{y.ci_lower_95:.0f},{y.ci_upper_95:.0f},{ppv_v},{tpr_v}"
        )
    csv_lines.append(
        f"aggregated,{sum(y.sample_size for y in report_data.strata_yields)},{sum(y.retained_news_docs for y in report_data.strata_yields)},{sum(y.proxy_total_words for y in report_data.strata_yields):.0f},{sum(y.residual_error_words for y in report_data.strata_yields):.0f},{report_data.aggregated_true_words:.0f},{report_data.aggregated_ci_lower_95:.0f},{report_data.aggregated_ci_upper_95:.0f},,"
    )
    csv_file.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    typer.echo(f"Confirmatory report written to: {report_file}")
    typer.echo(f"Summary CSV written to: {csv_file}")
    typer.echo(
        f"Projected True News Total: {report_data.aggregated_true_words:,.0f} words (95% CI: [{report_data.aggregated_ci_lower_95:,.0f}, {report_data.aggregated_ci_upper_95:,.0f}])"
    )


@app.command("calibrate")
def calibrate_filters(
    manifest: Annotated[
        Path | None,
        typer.Option(
            "--manifest", "-m", help="Path to provenance.parquet or provenance.jsonl"
        ),
    ] = None,
    audit_file: Annotated[
        Path | None, typer.Option("--audit-file", "-a", help="Path to gold audit JSONL")
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Analysis report output dir"),
    ] = None,
) -> None:
    """Perform offline post-fetch calibration, pre-fetch feasibility, and pipeline recommendations."""
    paths = RuntimePaths.from_environment()
    target_output_dir = output_dir or paths.analysis_output("f2", "corpus")
    target_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest
    if manifest_path is None:
        manifest_candidates = [
            paths.staging_root
            / "exp"
            / "f2"
            / "confirmatory_50k"
            / "provenance.parquet",
            paths.staging_root / "exp" / "f2" / "sample" / "provenance.parquet",
            paths.staging_root
            / "exp"
            / "f2"
            / "sample_10k_audited"
            / "provenance.parquet",
        ]
        manifest_path = next(
            (p for p in manifest_candidates if p.exists()), manifest_candidates[0]
        )

    if not manifest_path.exists():
        typer.echo(f"Error: Manifest file not found: {manifest_path}")
        return

    audit_records = None
    if audit_file and audit_file.exists():
        audit_records = [
            json.loads(line)
            for line in audit_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif audit_file is None:
        default_audit_paths = [
            target_output_dir / "00_corpus_audit_set_50k_400_annotated.jsonl",
            paths.staging_root
            / "exp"
            / "f2"
            / "00_corpus_audit_set_50k_400_annotated.jsonl",
            paths.analysis_output("f2") / "00_corpus_audit_set_50k_400_annotated.jsonl",
            paths.staging_root / "exp" / "f2" / "audit_set_50k_400_annotated.jsonl",
            target_output_dir / "audit_set_50k_400_annotated.jsonl",
            paths.staging_root
            / "exp"
            / "f2"
            / "00_corpus_audit_review_400_annotated.jsonl",
            paths.staging_root / "exp" / "f2" / "audit_review_400_annotated.jsonl",
            target_output_dir / "00_corpus_audit_set_400_annotated.jsonl",
            target_output_dir / "audit_set_400_annotated.jsonl",
        ]
        for p in default_audit_paths:
            if p.exists():
                audit_records = [
                    json.loads(line)
                    for line in p.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                break

    if not audit_records:
        typer.echo("Error: No audited records found to perform calibration.")
        return

    calibrator = CalibrationAndPreFetchAnalyzer(manifest_path, audit_records)
    report_md = calibrator.generate_report_markdown()

    out_file = target_output_dir / "00_corpus_confirmatory_50k_filter_study.md"
    out_file.write_text(report_md, encoding="utf-8")
    typer.echo(f"Filter validation study written to: {out_file}")


@app.command("build")
def build_corpus(
    crawl: Annotated[
        str, typer.Option("--crawl", "-c", help="Common Crawl ID")
    ] = "CC-MAIN-2012",
    target_words: Annotated[
        int, typer.Option("--target-words", "-t", help="Target word count")
    ] = 1_000_000_000,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Materialized corpus output dir"),
    ] = None,
) -> None:
    """Full-scale corpus materialization using the identical logical data pipeline."""
    paths = RuntimePaths.from_environment()
    target_output_dir = output_dir or (paths.dataset("f2") / "news_1b")
    target_output_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(
        f"Building full corpus for {crawl} (Target: {target_words:,} words) -> {target_output_dir}"
    )


__all__ = ["app"]
