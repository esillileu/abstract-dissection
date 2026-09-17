"""Deterministic acquisition and record-preserving corpus transformations."""

from __future__ import annotations

from .extraction import (
    extract_gigaword_documents,
    extract_wikipedia_records,
    iter_tar_records,
    iter_text_lines,
    open_source_records,
)
from .normalization import normalize_text
from .sharding import (
    DeterministicSharder,
    ShardInfo,
    iter_shard_text,
    open_canonical_shards,
)

__all__ = [
    "DeterministicSharder",
    "ShardInfo",
    "extract_gigaword_documents",
    "extract_wikipedia_records",
    "iter_shard_text",
    "iter_tar_records",
    "iter_text_lines",
    "normalize_text",
    "open_canonical_shards",
    "open_source_records",
]
