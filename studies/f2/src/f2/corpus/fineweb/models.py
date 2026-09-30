"""Data models and record contracts for FineWeb 2013 news surrogate corpus."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FineWebDocument:
    """Raw document extracted from FineWeb parquet with full source provenance."""

    fineweb_id: str
    text: str
    dump: str
    url: str
    date: str
    file_path: str | None
    language: str
    language_score: float
    token_count: int
    parquet_path: str
    row_index: int

    @property
    def canonical_id(self) -> str:
        """Stable, deterministic canonical document identifier."""
        stem = self.parquet_path.rsplit("/", 1)[-1].replace(".parquet", "")
        return f"fineweb:{self.dump}:{stem}:{self.row_index:07d}:{self.fineweb_id}"


@dataclass(frozen=True)
class FineWebNewsRecord:
    """Document accepted by the news filter and normalized for Word2Vec."""

    canonical_id: str
    normalized_text: str
    word_count: int
    news_score: float
    doc: FineWebDocument


@dataclass(frozen=True)
class FineWebProvenanceRecord:
    """Serializable provenance entry for an accepted surrogate news document."""

    canonical_id: str
    fineweb_id: str
    dataset: str
    revision: str
    dump: str
    parquet_file: str
    row_index: int
    url: str
    date: str
    word_count: int
    news_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "FineWebDocument",
    "FineWebNewsRecord",
    "FineWebProvenanceRecord",
]
