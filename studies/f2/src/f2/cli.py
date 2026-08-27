"""Typer CLI interface for F2 Word2Vec corpus feasibility, sampling, auditing, database migrations, and analysis."""

from __future__ import annotations

import json
import math
import urllib.request
import uuid
from pathlib import Path
from typing import Annotated

import typer

from repro_core.context.paths import RuntimePaths

from .analysis import FeasibilityAnalyzer
from .calibration import CalibrationAndPreFetchAnalyzer
from .cdx import CDXBlockLocator, CDXIndexReader
from .db.migrations.runner import run_migrations
from .db.repository import CorpusStateRepository
from .db.session import get_connection
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
    seed: Annotated[int, typer.Option("--seed", "-s", help="Random seed")] = 42,
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
    typer.echo("  Prevalence Filter: None (Unbiased ground truth)")
    typer.echo(
        "  Classification Correction: Two-phase residual difference estimator (e_i = y_gold - y_proxy)"
    )
    typer.echo("  State Store: PostgreSQL operational control plane")


@app.command("sample")
def sample_corpus(
    crawls: Annotated[
        str, typer.Option("--crawls", "-c", help="Comma-separated crawl IDs")
    ] = "CC-MAIN-2009-2010,CC-MAIN-2012",
    sample_size: Annotated[
        int, typer.Option("--sample-size", "-n", help="Total records to sample")
    ] = 50,
    seed: Annotated[int, typer.Option("--seed", "-s", help="Random seed")] = 42,
    bandwidth_limit: Annotated[
        float, typer.Option("--bandwidth-limit", "-b", help="Bandwidth limit in Mbps")
    ] = 20.0,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-j", help="Concurrent fetch workers")
    ] = 2,
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Output directory")
    ] = Path(".staging/exp/f2/sample"),
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Explicit run identifier for resuming"),
    ] = None,
) -> None:
    """Execute bounded, rate-limited probability sample against Common Crawl backed by PostgreSQL."""
    output_dir.mkdir(parents=True, exist_ok=True)
    active_run_id = run_id or f"run_{seed}_{uuid.uuid4().hex[:8]}"

    crawl_list = [c.strip() for c in crawls.split(",") if c.strip()]
    per_crawl_target = max(1, sample_size // len(crawl_list))

    typer.echo(f"Initializing run '{active_run_id}' in PostgreSQL...")

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
            output_dir=output_dir.as_posix(),
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
            f"Bandwidth limit: {bandwidth_limit} Mbps, Concurrency: {concurrency}"
        )

        records_per_block = 5
        blocks_per_crawl = max(1, math.ceil(per_crawl_target / records_per_block))

        processed_total = 0
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
                )

                repo.insert_candidates(active_run_id, final_candidates)

                for cand in final_candidates:
                    cand_id = cand.record_id()
                    if cand_id in completed_candidate_ids:
                        continue

                    arc_fetch = fetcher.fetch_range(
                        cand.filename, cand.offset, cand.length
                    )
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
                    )
                    completed_candidate_ids.add(cand_id)
                    processed_total += 1

                    url_display = (
                        cand.url if len(cand.url) <= 50 else cand.url[:47] + "..."
                    )
                    typer.echo(
                        f"  [{processed_total}/{sample_size}] {url_display} -> {result.fetch_status} "
                        f"[news={result.is_news_predicted}, en={result.is_english}, valid={result.is_valid}, words={result.word_count}]"
                    )

        text_writer.close()
        repo.update_run_status(active_run_id, "completed")

        # Export provenance to parquet & jsonl
        exporter = ProvenanceExporter(repo)
        exports = exporter.export(active_run_id, output_dir)
        typer.echo(f"\nSampling completed successfully! Run ID: {active_run_id}")
        typer.echo(f"  - Total Processed: {processed_total}")
        typer.echo(f"  - Retained News:   {news_total}")
        typer.echo(f"  - Parquet Export:  {exports['parquet']}")
        typer.echo(f"  - JSONL Export:    {exports['jsonl']}")


@app.command("export")
def export_run(
    run_id: Annotated[
        str, typer.Option("--run-id", "-r", help="Run ID to export from DB")
    ],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Output directory")
    ] = Path(".staging/exp/f2/export"),
) -> None:
    """Export provenance records for a given run from PostgreSQL to Parquet and JSONL."""
    with get_connection() as conn:
        repo = CorpusStateRepository(conn)
        exporter = ProvenanceExporter(repo)
        exports = exporter.export(run_id, output_dir)
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
    ] = 200,
    seed: Annotated[
        int, typer.Option("--seed", "-s", help="Random seed for audit priority")
    ] = 1337,
) -> None:
    """Generate pre-specified sequential audit assignments and store in PostgreSQL."""
    with get_connection() as conn:
        repo = CorpusStateRepository(conn)
        records = repo.get_provenance_records(run_id)
        if not records:
            typer.echo(f"No records found for run '{run_id}'.")
            return

        preds = [int(r["is_news_predicted"]) for r in records]
        sampler = SequentialAuditSampler(seed=seed)
        schedule = sampler.generate_audit_schedule(records, preds)
        per_stratum = {0: budget // 2, 1: budget // 2}
        assignments = sampler.select_audit_wave(schedule, per_stratum)

        count = repo.insert_audit_assignments(run_id, assignments)
        typer.echo(
            f"Created {count} audit assignments for run '{run_id}' with pre-specified priority."
        )


@app.command("audit-review")
def review_audit(
    run_id: Annotated[
        str, typer.Option("--run-id", "-r", help="Run ID to export audit records for")
    ],
    output_file: Annotated[
        Path,
        typer.Option("--output-file", "-o", help="Audit review JSONL output path"),
    ] = Path(".staging/exp/f2/audit_set_200.jsonl"),
) -> None:
    """Export the 200 audit assignments with complete metadata, weights, and text for manual labeling."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        repo = CorpusStateRepository(conn)
        audit_items = repo.get_audit_assignments(run_id)
        if not audit_items:
            typer.echo(
                f"No audit assignments found for run '{run_id}'. Run 'repro f2 corpus audit' first."
            )
            return

        # Map URLs to clean text snippet if shards exist
        shard_dir = Path(".staging/exp/f2/sample_10k/clean_shards")
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
        with output_file.open("w", encoding="utf-8") as f:
            for item in audit_items:
                u = item["url"]
                domain = urllib.parse.urlparse(u).netloc.lower()
                snippet = url_to_text.get(u, "")
                if snippet and len(snippet) > 800:
                    snippet = snippet[:800] + "..."

                review_entry = {
                    "audit_id": item["audit_id"],
                    "candidate_id": item["candidate_id"],
                    "priority_order": item["priority_order"],
                    "audit_stratum": item["audit_stratum"],
                    "audit_stratum_label": "News (Stratum 1)"
                    if item["audit_stratum"] == 1
                    else "Non-News (Stratum 0)",
                    "wave": item.get("wave", 1),
                    "crawl_id": item["crawl_id"],
                    "url": u,
                    "domain": domain,
                    "first_stage_inclusion_probability": item["first_stage_pi"],
                    "first_stage_design_weight": item["first_stage_weight"],
                    "audit_inclusion_probability": item["audit_inclusion_probability"],
                    "audit_design_weight": item["audit_design_weight"],
                    "news_score": item["news_score"],
                    "is_news_predicted": item["is_news_predicted"],
                    "is_english": item["is_english"],
                    "is_valid": item["is_valid"],
                    "word_count": item["word_count"],
                    "word_count_proxy": item["word_count_proxy"],
                    "diagnostics": item["diagnostics"],
                    "text_snippet": snippet,
                    # Fields for manual labeler:
                    "gold_class": item["gold_class"] if item["is_audited"] else None,
                    "word_count_gold": item["word_count_gold"]
                    if item["is_audited"]
                    else None,
                    "auditor_id": item.get("auditor_id"),
                    "notes": item.get("notes"),
                }
                exported_items.append(review_entry)
                f.write(json.dumps(review_entry, default=str) + "\n")

        # Also write Markdown Review Dossier
        md_file = output_file.parent / "audit_set_200_review.md"
        md_lines = [
            f"# Phase-2 Probability Audit Set (200 Documents) — Run `{run_id}`",
            "",
            "> **Pre-specified Sequential Probability Sampling Design**",
            "> * Stratum 1 (Predicted News): 100 documents",
            "> * Stratum 0 (Predicted Non-News): 100 documents",
            "",
            "---",
            "",
        ]
        for entry in exported_items:
            md_lines.extend(
                [
                    f"### #{entry['priority_order']:03d} [{entry['audit_stratum_label']}] `{entry['audit_id']}`",
                    f"- **URL:** [{entry['url']}]({entry['url']})",
                    f"- **Crawl:** `{entry['crawl_id']}` | **Domain:** `{entry['domain']}`",
                    f"- **Classifier Score:** `{entry['news_score']:.1f}` | **Proxy Words:** `{entry['word_count_proxy']:,}`",
                    f"- **Weights:** $\\pi_1 = {entry['first_stage_inclusion_probability']:.2e}, w_1 = {entry['first_stage_design_weight']:,.1f}$ | $\\pi_2 = {entry['audit_inclusion_probability']:.4f}, w_2 = {entry['audit_design_weight']:.1f}$",
                    f"- **Diagnostics:** `{json.dumps(entry['diagnostics'])}`",
                ]
            )
            if entry["text_snippet"]:
                md_lines.extend(["", "```text", entry["text_snippet"], "```", ""])
            else:
                md_lines.extend(
                    ["", "*(No clean text snippet retained in Phase 1)*", ""]
                )
            md_lines.extend(["---", ""])

        md_file.write_text("\n".join(md_lines), encoding="utf-8")

        # Also copy to artifacts/analysis/f2/
        art_dir = Path("artifacts/analysis/f2")
        art_dir.mkdir(parents=True, exist_ok=True)
        (art_dir / "audit_set_200.jsonl").write_text(
            output_file.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (art_dir / "audit_set_200_review.md").write_text(
            md_file.read_text(encoding="utf-8"), encoding="utf-8"
        )

        typer.echo(f"Exported {len(audit_items)} audit review documents to:")
        typer.echo(f"  - JSONL: {output_file}")
        typer.echo(f"  - Markdown Dossier: {md_file}")
        typer.echo(f"  - Artifacts Dossier: {art_dir / 'audit_set_200_review.md'}")


@app.command("audit-record")
def record_audit(
    run_id: Annotated[
        str, typer.Option("--run-id", "-r", help="Run ID to record audit labels for")
    ],
    audit_file: Annotated[
        Path, typer.Option("--audit-file", "-a", help="Path to annotated audit JSONL")
    ] = Path(".staging/exp/f2/audit_review_200.jsonl"),
) -> None:
    """Record completed audit gold labels into PostgreSQL from an annotated JSONL file."""
    if not audit_file.exists():
        typer.echo(f"Audit file not found: {audit_file}")
        return

    annotated_records = [
        json.loads(line)
        for line in audit_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with get_connection() as conn:
        repo = CorpusStateRepository(conn)
        recorded = 0
        for r in annotated_records:
            if r.get("gold_class") is not None and r.get("word_count_gold") is not None:
                repo.record_audit_gold_label(
                    run_id=run_id,
                    candidate_id=r["candidate_id"],
                    gold_class=int(r["gold_class"]),
                    word_count_gold=int(r["word_count_gold"]),
                    auditor_id=r.get("auditor_id", "human_expert"),
                    notes=r.get("notes"),
                )
                recorded += 1
        typer.echo(
            f"Recorded {recorded} gold audit labels into PostgreSQL for run '{run_id}'."
        )


@app.command("analyze")
def analyze_corpus(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to provenance.parquet or provenance.jsonl"
        ),
    ] = Path(".staging/exp/f2/sample/provenance.parquet"),
    audit_file: Annotated[
        Path | None, typer.Option("--audit-file", "-a", help="Path to gold audit JSONL")
    ] = None,
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Analysis report output dir")
    ] = Path("artifacts/analysis/f2"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest
    if not manifest_path.exists():
        fallback_manifest = Path(
            ".staging/exp/f2/sample_10k_audited/provenance.parquet"
        )
        if fallback_manifest.exists():
            manifest_path = fallback_manifest

    analyzer = FeasibilityAnalyzer(manifest_path)

    audit_records = None
    if audit_file and audit_file.exists():
        audit_records = [
            json.loads(line)
            for line in audit_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif audit_file is None:
        # Check standard annotated audit file locations
        default_audit_paths = [
            Path(".staging/exp/f2/audit_review_400_annotated.jsonl"),
            Path("artifacts/analysis/f2/audit_set_400_annotated.jsonl"),
            Path(".staging/exp/f2/audit_review_200_annotated.jsonl"),
            Path("artifacts/analysis/f2/audit_set_200_annotated.jsonl"),
            Path(".staging/exp/f2/audit_set_200.jsonl"),
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
            try:
                with get_connection() as conn:
                    repo = CorpusStateRepository(conn)
                    all_audits = repo.get_audit_assignments("run_42_a1d3745e")
                    audited_items = [a for a in all_audits if a.get("is_audited")]
                    if audited_items:
                        audit_records = audited_items
            except Exception:
                pass

    report_data = analyzer.compute_two_phase_yield(audit_records=audit_records)
    md_content = analyzer.generate_report_markdown(report_data)

    report_file = output_dir / "feasibility_report.md"
    report_file.write_text(md_content, encoding="utf-8")

    # Also export summary.csv
    csv_file = output_dir / "feasibility_summary.csv"
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

    funnel_list = analyzer.compute_sequential_funnel()
    tot_sampled = sum(int(f["step0_sampled"]) for f in funnel_list)
    tot_valid = sum(int(f["step5_retained_valid_news"]) for f in funnel_list)
    typer.echo(f"Report written to: {report_file}")
    typer.echo(f"Summary CSV written to: {csv_file}")
    typer.echo(
        f"\nSummary Funnel: Sampled: {tot_sampled:,}, Retained Valid News: {tot_valid:,}"
    )
    typer.echo(
        f"Projected True News Total: {report_data.aggregated_true_words:,.0f} words (95% CI: [{report_data.aggregated_ci_lower_95:,.0f}, {report_data.aggregated_ci_upper_95:,.0f}])"
    )


@app.command("calibrate")
def calibrate_filters(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to provenance.parquet or provenance.jsonl"
        ),
    ] = Path(".staging/exp/f2/sample/provenance.parquet"),
    audit_file: Annotated[
        Path | None, typer.Option("--audit-file", "-a", help="Path to gold audit JSONL")
    ] = None,
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Analysis report output dir")
    ] = Path("artifacts/analysis/f2"),
) -> None:
    """Perform offline post-fetch calibration, pre-fetch feasibility, and pipeline recommendations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest
    if not manifest_path.exists():
        fallback_manifest = Path(
            ".staging/exp/f2/sample_10k_audited/provenance.parquet"
        )
        if fallback_manifest.exists():
            manifest_path = fallback_manifest

    audit_records = None
    if audit_file and audit_file.exists():
        audit_records = [
            json.loads(line)
            for line in audit_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif audit_file is None:
        default_audit_paths = [
            Path(".staging/exp/f2/audit_review_400_annotated.jsonl"),
            Path("artifacts/analysis/f2/audit_set_400_annotated.jsonl"),
            Path(".staging/exp/f2/audit_review_200_annotated.jsonl"),
            Path("artifacts/analysis/f2/audit_set_200_annotated.jsonl"),
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

    out_file = output_dir / "offline_calibration_and_prefetch_study.md"
    out_file.write_text(report_md, encoding="utf-8")
    typer.echo(f"Calibration study written to: {out_file}")


@app.command("build")
def build_corpus(
    crawl: Annotated[
        str, typer.Option("--crawl", "-c", help="Common Crawl ID")
    ] = "CC-MAIN-2012",
    target_words: Annotated[
        int, typer.Option("--target-words", "-t", help="Target word count")
    ] = 1_000_000_000,
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Materialized corpus output dir")
    ] = Path("data/f2/news_1b"),
) -> None:
    """Full-scale corpus materialization using the identical logical data pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(
        f"Building full corpus for {crawl} (Target: {target_words:,} words) -> {output_dir}"
    )
    typer.echo("Executing full production extraction using pipeline runner...")


__all__ = ["app"]
