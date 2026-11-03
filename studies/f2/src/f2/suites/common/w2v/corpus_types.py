"""Corpus binding data models and protocols."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ObjectReader(Protocol):
    """The read-only part of ``repro_io.s3.S3ObjectStore`` used here."""

    def get_file(self, uri: str, local_path: Path) -> Path: ...


@dataclass(frozen=True)
class CorpusShard:
    index: int
    uri: str
    sha256: str
    byte_size: int
    word_count: int
    document_count: int


@dataclass(frozen=True)
class CorpusBinding:
    shards: tuple[CorpusShard, ...]

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Mapping[str, object]],
    ) -> CorpusBinding:
        """Translate verified corpus DB rows into ordered shard inputs."""
        return cls(
            tuple(
                CorpusShard(
                    index=int(row["shard_index"]),
                    uri=str(row["s3_uri"]),
                    sha256=str(row["sha256"]),
                    byte_size=int(row["byte_size"]),
                    word_count=int(row["word_count"]),
                    document_count=int(row["doc_count"]),
                )
                for row in rows
            )
        )


@dataclass(frozen=True)
class MaterializedCorpus:
    path: Path
    corpus_sha256: str
    lexical_tokens: int
    complete_shards: int


__all__ = [
    "CorpusBinding",
    "CorpusShard",
    "MaterializedCorpus",
    "ObjectReader",
]
