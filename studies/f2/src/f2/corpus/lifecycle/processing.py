"""Canonical and normalized corpus extraction and sharding pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..canonical import (
    DeterministicSharder,
    ShardInfo,
    normalize_text,
    open_canonical_shards,
    open_source_records,
)
from ..sources import SOURCE_BOUNDARY_POLICIES, CorpusSource


def process_canonical_source(
    source: CorpusSource,
    raw_inputs: list[Path],
    output_dir: Path,
    *,
    target_words: int = 10_000_000,
) -> tuple[list[ShardInfo], Path]:
    """Extract visible/released text from raw archives without lowercase or token normalization."""
    records = open_source_records(source.key, raw_inputs)
    policy = SOURCE_BOUNDARY_POLICIES.get(source.key, "sentence_per_line")
    sharder = DeterministicSharder(output_dir, target_words)
    shards = sharder.write(records, source=source.key, boundary_policy=policy)
    return shards, output_dir / "manifest.json"


def process_normalized_source(
    source: CorpusSource,
    canonical_shard_paths: list[Path],
    output_dir: Path,
    *,
    target_words: int = 10_000_000,
) -> tuple[list[ShardInfo], Path]:
    """Apply Mikolov demo-train-big-model-v1.sh normalize_text() to canonical shards."""
    canonical_records = open_canonical_shards(canonical_shard_paths)
    records = ((rec_id, normalize_text(text)) for rec_id, text in canonical_records)
    policy = SOURCE_BOUNDARY_POLICIES.get(source.key, "sentence_per_line")
    sharder = DeterministicSharder(output_dir, target_words)
    shards = sharder.write(records, source=source.key, boundary_policy=policy)
    return shards, output_dir / "manifest.json"


def process_source(
    source: CorpusSource,
    inputs: list[Path],
    output_dir: Path,
    *,
    normalized: bool = True,
    target_words: int = 10_000_000,
) -> tuple[list[Any], Path]:
    if normalized:
        records = open_source_records(source.key, inputs)
        norm_records = ((ident, normalize_text(text)) for ident, text in records)
        policy = SOURCE_BOUNDARY_POLICIES.get(source.key, "sentence_per_line")
        sharder = DeterministicSharder(output_dir, target_words)
        shards = sharder.write(norm_records, source=source.key, boundary_policy=policy)
        return shards, output_dir / "manifest.json"
    return process_canonical_source(
        source, inputs, output_dir, target_words=target_words
    )


__all__ = [
    "process_canonical_source",
    "process_normalized_source",
    "process_source",
]
