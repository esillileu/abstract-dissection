"""Frozen production specification for the FineWeb 2013 news surrogate corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .adapter import (
    FINEWEB_DATASET,
    FINEWEB_DEFAULT_DUMP,
    FINEWEB_PINNED_REVISION,
)


@dataclass(frozen=True)
class SourceIdentity:
    dataset: str = FINEWEB_DATASET
    pinned_revision: str = FINEWEB_PINNED_REVISION
    primary_dump: str = FINEWEB_DEFAULT_DUMP
    fallback_dump: str | None = None
    ordering: str = "lexicographical_parquet_asc_row_index_asc"


@dataclass(frozen=True)
class FilterIdentity:
    classifier: str = "FineWebNewsClassifier-v1"
    min_score: float = 1.5
    min_raw_words: int = 100
    language: str = "en"
    min_language_score: float = 0.65


@dataclass(frozen=True)
class PreprocessingIdentity:
    recipe: str = "mikolov_demo_train_big_model_v1_normalize_text"
    locale: str = "C"
    word_definition: str = "non_whitespace_token_split"
    sentence_boundary_token: str = "</s>"
    sentence_boundary_char: str = "\n"
    boundary_policy: str = "document_paragraph_lines"


@dataclass(frozen=True)
class SelectionIdentity:
    target_33b_words: int = 33_000_000_000
    target_6b_words: int = 6_000_000_000
    subset_relationship: str = "C_6B is exact deterministic prefix of C_33B"
    word_boundary_policy: str = "exact_token_slice_at_budget"


@dataclass(frozen=True)
class OutputIdentity:
    shard_pattern: str = "shard-{index:05d}.txt.zst"
    target_words_per_shard: int = 10_000_000
    compression: str = "zstd -q -f -T1 -19 --no-progress"
    provenance_file: str = "provenance.jsonl"
    manifest_file: str = "manifest.json"
    checkpoint_file: str = "checkpoint.json"


@dataclass(frozen=True)
class ProductionCorpusSpec:
    """Master immutable production specification for FineWeb 2013 news surrogate."""

    spec_version: int = 1
    name: str = "fineweb-2013-news-surrogate"
    source: SourceIdentity = SourceIdentity()
    filter: FilterIdentity = FilterIdentity()
    preprocessing: PreprocessingIdentity = PreprocessingIdentity()
    selection: SelectionIdentity = SelectionIdentity()
    output: OutputIdentity = OutputIdentity()

    @property
    def spec_hash(self) -> str:
        """Deterministic SHA-256 digest of the complete configuration payload."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRODUCTION_SPEC = ProductionCorpusSpec()

__all__ = [
    "PRODUCTION_SPEC",
    "FilterIdentity",
    "OutputIdentity",
    "PreprocessingIdentity",
    "ProductionCorpusSpec",
    "SelectionIdentity",
    "SourceIdentity",
]
