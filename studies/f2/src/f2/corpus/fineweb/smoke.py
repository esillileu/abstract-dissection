"""Smoke test runner for FineWeb source adapter and end-to-end processing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa

from .adapter import FINEWEB_DEFAULT_DUMP, FineWebSourceAdapter
from .pipeline import FineWebPipeline
from .writer import FineWebShardWriter


def run_fineweb_smoke(
    output_dir: Path,
    *,
    sample_size: int = 50,
    min_words: int = 50,
    dump: str = FINEWEB_DEFAULT_DUMP,
    arrow_table: pa.Table | None = None,
    remote: bool = False,
    target_words_per_shard: int = 5_000,
) -> dict[str, Any]:
    """Execute a minimal smoke test through the FineWeb adapter, filter, and writer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = FineWebSourceAdapter()
    pipeline = FineWebPipeline(min_words=min_words)
    writer = FineWebShardWriter(
        output_dir,
        target_words=target_words_per_shard,
        dump=dump,
    )

    if arrow_table is not None:
        docs = adapter.iter_arrow_table(arrow_table, dump=dump, limit=sample_size)
    elif remote:
        first_file = adapter.list_parquet_files(dump)[0]
        url = adapter.resolve_url(first_file)
        docs = adapter.iter_remote_parquet(url, dump=dump, limit=sample_size)
    else:
        raise ValueError("Must provide either arrow_table or remote=True")

    accepted_docs = 0
    total_words = 0

    for news_rec, prov_rec in pipeline.process_stream(docs):
        writer.append(news_rec, prov_rec)
        accepted_docs += 1
        total_words += news_rec.word_count

    shards = writer.close()

    return {
        "documents_seen": pipeline.stats.documents_seen,
        "documents_accepted": accepted_docs,
        "total_raw_words": pipeline.stats.total_raw_words,
        "total_accepted_words": total_words,
        "acceptance_ratio": (
            accepted_docs / pipeline.stats.documents_seen
            if pipeline.stats.documents_seen > 0
            else 0.0
        ),
        "shards_written": len(shards),
        "manifest_path": writer.manifest_path,
        "provenance_path": writer.provenance_path,
        "shards": shards,
    }


__all__ = ["run_fineweb_smoke"]
