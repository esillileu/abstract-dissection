from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from repro_core.context.paths import RuntimePaths

from ..analysis import FeasibilityAnalyzer
from ..calibration import CalibrationAndPreFetchAnalyzer


def analyze_corpus(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to provenance.parquet or provenance.jsonl"
        ),
    ],
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
    target_output_dir = output_dir or paths.analysis_output("f2_cc", "corpus")
    target_output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest.exists():
        typer.echo(f"Error: Manifest file not found: {manifest}")
        return

    analyzer = FeasibilityAnalyzer(manifest)

    audit_records = None
    if audit_file and audit_file.exists():
        audit_records = [
            json.loads(line)
            for line in audit_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    report_data = analyzer.compute_two_phase_yield(
        audit_records=audit_records, bootstrap_reps=1000
    )
    md_content = analyzer.generate_report_markdown(report_data)

    report_file = target_output_dir / "corpus_analysis.md"
    report_file.write_text(md_content, encoding="utf-8")

    # Also export summary.csv
    csv_file = target_output_dir / "corpus_analysis.csv"
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


def calibrate_filters(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest", "-m", help="Path to provenance.parquet or provenance.jsonl"
        ),
    ],
    audit_file: Annotated[
        Path, typer.Option("--audit-file", "-a", help="Path to gold audit JSONL")
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Analysis report output dir"),
    ] = None,
) -> None:
    """Perform offline post-fetch calibration, pre-fetch feasibility, and pipeline recommendations."""
    paths = RuntimePaths.from_environment()
    target_output_dir = output_dir or paths.analysis_output("f2_cc", "corpus")
    target_output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest.exists():
        typer.echo(f"Error: Manifest file not found: {manifest}")
        return

    audit_records = None
    if audit_file.exists():
        audit_records = [
            json.loads(line)
            for line in audit_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    if not audit_records:
        typer.echo("Error: No audited records found to perform calibration.")
        return

    calibrator = CalibrationAndPreFetchAnalyzer(manifest, audit_records)
    report_md = calibrator.generate_report_markdown()

    out_file = target_output_dir / "filter_calibration.md"
    out_file.write_text(report_md, encoding="utf-8")
    typer.echo(f"Filter validation study written to: {out_file}")
