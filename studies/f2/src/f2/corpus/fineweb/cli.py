"""CLI commands for FineWeb 2013 surrogate corpus."""

from __future__ import annotations

from pathlib import Path

import typer

from repro_core.context.paths import RuntimePaths

from .adapter import (
    FINEWEB_DATASET,
    FINEWEB_DEFAULT_DUMP,
    FINEWEB_PINNED_REVISION,
    FineWebSourceAdapter,
)
from .smoke import run_fineweb_smoke

fineweb_app = typer.Typer(
    name="fineweb",
    help="FineWeb 2013 news surrogate corpus acquisition and processing.",
    no_args_is_help=True,
)


@fineweb_app.command("info")
def fineweb_info(
    dump: str = typer.Option(FINEWEB_DEFAULT_DUMP, help="Common Crawl dump ID"),
) -> None:
    """Display FineWeb source configuration and pinned revision."""
    adapter = FineWebSourceAdapter()
    files = adapter.list_parquet_files(dump)
    typer.echo(f"FineWeb Dataset : {FINEWEB_DATASET}")
    typer.echo(f"Pinned Revision : {FINEWEB_PINNED_REVISION}")
    typer.echo(f"Dump            : {dump}")
    typer.echo(f"Total Parquet   : {len(files)} files")
    typer.echo(f"First File URL  : {adapter.resolve_url(files[0])}")


@fineweb_app.command("smoke")
def fineweb_smoke(
    sample_size: int = typer.Option(50, help="Number of documents to sample"),
    dump: str = typer.Option(FINEWEB_DEFAULT_DUMP, help="Dump identifier"),
    min_words: int = typer.Option(
        100, help="Minimum raw words for candidate documents"
    ),
    output_dir: Path | None = typer.Option(
        None, help="Output directory for smoke shards and manifest"
    ),
) -> None:
    """Execute an end-to-end smoke run on remote FineWeb parquet data."""
    paths = RuntimePaths.from_environment()
    target_dir = output_dir or paths.staging_root / "exp" / "f2" / "fineweb" / "smoke"

    typer.echo(
        f"Running FineWeb smoke test (sample_size={sample_size}, dump={dump})..."
    )
    res = run_fineweb_smoke(
        target_dir,
        sample_size=sample_size,
        min_words=min_words,
        dump=dump,
        remote=True,
    )

    typer.echo("--- Smoke Test Results ---")
    typer.echo(f"Documents Seen     : {res['documents_seen']}")
    typer.echo(f"Documents Accepted : {res['documents_accepted']}")
    typer.echo(f"Acceptance Ratio   : {res['acceptance_ratio']:.2%}")
    typer.echo(f"Total Raw Words    : {res['total_raw_words']}")
    typer.echo(f"Accepted Words     : {res['total_accepted_words']}")
    typer.echo(f"Shards Written     : {res['shards_written']}")
    typer.echo(f"Manifest Path      : {res['manifest_path']}")
    typer.echo(f"Provenance Path    : {res['provenance_path']}")


@fineweb_app.command("feasibility")
def fineweb_feasibility(
    sample_size: int = typer.Option(
        1000, help="Number of documents to sample for feasibility"
    ),
    dump: str = typer.Option(FINEWEB_DEFAULT_DUMP, help="Dump identifier"),
    min_words: int = typer.Option(
        100, help="Minimum raw words for candidate documents"
    ),
) -> None:
    """Execute feasibility study and yield estimation for 33B corpus."""
    from .feasibility import run_feasibility_study

    typer.echo(
        f"Running FineWeb feasibility study (sample_size={sample_size}, dump={dump})..."
    )
    report = run_feasibility_study(
        sample_size=sample_size, dump=dump, min_words=min_words
    )

    typer.echo("\n================ FEASIBILITY STUDY RESULTS ================")
    typer.echo(f"Dump Sampled                : {report.dump}")
    typer.echo(f"Documents Seen              : {report.documents_seen:,}")
    typer.echo(f"Source Words Seen           : {report.source_words_seen:,}")
    typer.echo(f"News Documents Accepted     : {report.news_documents_accepted:,}")
    typer.echo(f"Accepted W2V Words          : {report.accepted_w2v_words:,}")
    typer.echo(f"Document Acceptance Ratio   : {report.document_acceptance_ratio:.2%}")
    typer.echo(f"Word Acceptance Ratio       : {report.word_acceptance_ratio:.2%}")
    typer.echo(
        f"Mean Source Doc Length      : {report.mean_source_doc_length:.1f} words"
    )
    typer.echo(
        f"Mean Accepted Doc Length    : {report.mean_accepted_doc_length:.1f} words"
    )
    typer.echo(
        f"Median Accepted Doc Length  : {report.median_accepted_doc_length:.1f} words"
    )
    typer.echo("---------------- THROUGHPUT BREAKDOWN ----------------")
    typer.echo(
        f"FineWeb Read Throughput     : {report.read_throughput_docs_sec:.1f} docs/s"
    )
    typer.echo(
        f"News Filter Throughput      : {report.filter_throughput_docs_sec:.1f} docs/s"
    )
    typer.echo(
        f"Preprocessing Throughput    : {report.prep_throughput_docs_sec:.1f} docs/s"
    )
    typer.echo(
        f"Total Processing Throughput : {report.total_throughput_docs_sec:.1f} docs/s"
    )
    typer.echo("---------------- 33B YIELD & CAPACITY ----------------")
    typer.echo(
        f"Source Words Req for 33B    : {report.source_words_required_for_33b:,}"
    )
    typer.echo(f"Source Docs Req for 33B     : {report.source_docs_required_for_33b:,}")
    typer.echo(f"Dump Total Documents        : {report.dump_total_documents:,}")
    typer.echo(f"Estimated Dump Total Words  : {report.estimated_dump_total_words:,}")
    typer.echo(
        f"Estimated Dump News Words   : {report.estimated_dump_accepted_words:,}"
    )
    typer.echo(
        f"Can Fulfill 33B with {report.dump}? : {'YES' if report.can_fulfill_33b_with_dump_20 else 'NO'}"
    )
    typer.echo(f"Margin Ratio over 33B       : {report.margin_ratio_20:.2f}x")
    typer.echo(
        f"Requires CC-MAIN-2013-48?   : {'YES' if report.requires_dump_48 else 'NO'}"
    )
    typer.echo(
        f"Estimated Compressed Storage: {report.estimated_storage_gb_33b:.2f} GB (zstd -19)"
    )
    typer.echo("===========================================================\n")


@fineweb_app.command("build")
def fineweb_build(
    target_words: int = typer.Option(
        33_000_000_000, help="Target W2V words to accumulate"
    ),
    target_words_per_shard: int = typer.Option(
        10_000_000, help="Target words per shard"
    ),
    dump: str = typer.Option(FINEWEB_DEFAULT_DUMP, help="Dump identifier"),
    min_words: int = typer.Option(
        100, help="Minimum raw words for candidate documents"
    ),
    peak_mbps: float = typer.Option(
        40.0, help="Bandwidth limit during peak hours (Mbps)"
    ),
    offpeak_mbps: float = typer.Option(
        100.0, help="Bandwidth limit during off-peak hours (Mbps)"
    ),
    no_limit: bool = typer.Option(False, help="Disable bandwidth throttling"),
    max_files: int | None = typer.Option(
        None, help="Maximum number of parquet files to process"
    ),
    output_dir: Path | None = typer.Option(
        None, help="Output directory for production shards, manifest, and checkpoint"
    ),
) -> None:
    """Execute production materialization with rate-limiting and checkpointing."""
    from .materializer import FineWebProductionMaterializer

    paths = RuntimePaths.from_environment()
    target_dir = output_dir or paths.staging_root / "exp" / "f2" / "fineweb" / "33b"

    materializer = FineWebProductionMaterializer(
        target_dir,
        target_words=target_words,
        target_words_per_shard=target_words_per_shard,
        peak_mbps=peak_mbps,
        offpeak_mbps=offpeak_mbps,
        no_bandwidth_limit=no_limit,
        min_words=min_words,
        dump=dump,
    )

    typer.echo(
        f"Starting FineWeb production materialization (target={target_words:,} words, dump={dump})..."
    )
    summary = materializer.run(max_parquet_files=max_files)

    typer.echo("\n--- Materialization Summary ---")
    typer.echo(
        f"Cumulative Words    : {summary['cumulative_words']:,} / {summary['target_words']:,}"
    )
    typer.echo(f"Documents Seen      : {summary['documents_seen']:,}")
    typer.echo(f"Documents Accepted  : {summary['documents_accepted']:,}")
    typer.echo(f"Shards Written      : {summary['shards_count']:,}")
    typer.echo(f"Files Processed     : {summary['files_processed']:,}")
    typer.echo(f"Elapsed Seconds     : {summary['elapsed_seconds']:.1f}s")
    typer.echo(f"Manifest Path       : {summary['manifest_path']}")
    typer.echo(f"Checkpoint Path     : {summary['checkpoint_path']}")


@fineweb_app.command("register")
def fineweb_register(
    checkpoint_path: Path = typer.Option(
        Path("/data/iso/f2-corpus/fineweb-33b/checkpoint.json"),
        help="Path to FineWeb checkpoint.json",
    ),
    shards_dir: Path = typer.Option(
        Path("/data/iso/f2-corpus/fineweb-33b/shards"),
        help="Directory containing shard files",
    ),
    max_shards: int | None = typer.Option(
        None,
        help="Maximum number of shards to register (default: all completed shards)",
    ),
    resource_id: str = typer.Option(
        "f2-fineweb-normalized",
        help="Catalog resource ID",
    ),
    resource_version_id: str = typer.Option(
        "f2-fineweb-2013-news-normalized-v1",
        help="Catalog resource version ID",
    ),
    version_label: str = typer.Option(
        "v1",
        help="Version label for catalog",
    ),
    upload_s3: bool = typer.Option(
        True,
        help="Upload shards to SeaweedFS S3 if not already present",
    ),
    register_plan: bool = typer.Option(
        True,
        help="Register W2V1 FineWeb execution plan and run slots in catalog",
    ),
    max_workers: int = typer.Option(
        8,
        help="Concurrent upload worker threads",
    ),
) -> None:
    """Register materialized FineWeb shards into SeaweedFS S3 and PostgreSQL catalog."""
    from repro_io.s3 import S3ObjectStore

    from f2.corpus.db.session import get_connection
    from f2.corpus.object_store import s3_config_from_environment

    from .register import (
        load_and_validate_checkpoint,
        register_corpus_in_db,
        register_execution_plan,
        upload_shards_to_s3,
    )

    shard_target = f"first {max_shards}" if max_shards else "all completed"
    typer.echo(
        f"Loading and validating {shard_target} shards from {checkpoint_path}..."
    )
    shards = load_and_validate_checkpoint(checkpoint_path, shards_dir, max_shards)
    total_words = sum(s["word_count"] for s in shards)
    total_bytes = sum(s["compressed_bytes"] for s in shards)
    typer.echo(
        f"Validated {len(shards)} shards: {total_words:,} words ({total_bytes / (1024**3):.2f} GB compressed)"
    )

    store = S3ObjectStore(s3_config_from_environment())
    if upload_s3:
        typer.echo("Syncing shards to SeaweedFS S3...")
        uploaded = upload_shards_to_s3(
            shards, shards_dir, store, max_workers=max_workers
        )
        typer.echo(
            f"S3 sync complete: {uploaded} newly uploaded shards (out of {len(shards)})"
        )

    typer.echo("Registering corpus in PostgreSQL catalog...")
    with get_connection() as conn:
        stats = register_corpus_in_db(
            shards,
            conn,
            store,
            resource_id=resource_id,
            resource_version_id=resource_version_id,
            version_label=version_label,
        )
        typer.echo(
            f"DB registration complete: {stats['shards']} shards, {stats['total_words']:,} words registered."
        )

        if register_plan:
            typer.echo("Registering W2V1 execution plan and run slots...")
            plan_info = register_execution_plan(
                conn, resource_version_id=resource_version_id
            )
            typer.echo(
                f"Plan registered: {plan_info['execution_plan_id']} ({plan_info['planned_slots_count']} slots)"
            )

    typer.echo("\n--- FineWeb Registration Summary ---")
    typer.echo(f"Resource ID         : {resource_id}")
    typer.echo(f"Resource Version ID : {resource_version_id}")
    typer.echo(f"Shards Registered   : {len(shards)}")
    typer.echo(f"Total Words         : {total_words:,}")
    typer.echo(f"Total Size          : {total_bytes / (1024**3):.2f} GB")
    typer.echo(f"S3 URI Base         : {store.uri('processed/fineweb/normalized/')}")
    typer.echo("Status              : READY FOR EXPERIMENTS")


__all__ = ["fineweb_app"]
