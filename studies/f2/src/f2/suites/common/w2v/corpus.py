"""Compatibility re-exports for refactored corpus binding and materialization modules."""

from __future__ import annotations

from .corpus_materializer import CorpusMaterializer
from .corpus_streaming import (
    CORPUS_CHUNK_BYTES,
    _copy_corpus_prefix,
    _decompressed_chunks,
    slice_corpus_file,
)
from .corpus_types import (
    CorpusBinding,
    CorpusShard,
    MaterializedCorpus,
    ObjectReader,
)

__all__ = [
    "CORPUS_CHUNK_BYTES",
    "CorpusBinding",
    "CorpusMaterializer",
    "CorpusShard",
    "MaterializedCorpus",
    "ObjectReader",
    "_copy_corpus_prefix",
    "_decompressed_chunks",
    "slice_corpus_file",
]
