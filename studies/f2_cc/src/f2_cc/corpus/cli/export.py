from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from repro_core.context.paths import RuntimePaths

from ...db.repository import CorpusStateRepository
from ...db.session import get_connection
from ..storage import ProvenanceExporter


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
