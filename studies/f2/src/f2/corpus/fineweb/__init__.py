"""FineWeb 2013 news surrogate corpus module for Word2Vec training."""

from __future__ import annotations

from .adapter import (
    FINEWEB_DATASET,
    FINEWEB_DEFAULT_DUMP,
    FINEWEB_FALLBACK_DUMP,
    FINEWEB_PINNED_REVISION,
    FINEWEB_TOTAL_PARQUET_FILES,
    FineWebSourceAdapter,
)
from .classifier import FineWebNewsClassifier
from .materializer import (
    FineWebProductionMaterializer,
    MaterializerCheckpoint,
)
from .models import (
    FineWebDocument,
    FineWebNewsRecord,
    FineWebProvenanceRecord,
)
from .pipeline import FineWebPipeline, PipelineStatistics
from .smoke import run_fineweb_smoke
from .spec import PRODUCTION_SPEC, ProductionCorpusSpec
from .writer import FineWebShardWriter

__all__ = [
    "FINEWEB_DATASET",
    "FINEWEB_DEFAULT_DUMP",
    "FINEWEB_FALLBACK_DUMP",
    "FINEWEB_PINNED_REVISION",
    "FINEWEB_TOTAL_PARQUET_FILES",
    "PRODUCTION_SPEC",
    "FineWebDocument",
    "FineWebNewsClassifier",
    "FineWebNewsRecord",
    "FineWebPipeline",
    "FineWebProductionMaterializer",
    "FineWebProvenanceRecord",
    "FineWebShardWriter",
    "FineWebSourceAdapter",
    "MaterializerCheckpoint",
    "PipelineStatistics",
    "ProductionCorpusSpec",
    "run_fineweb_smoke",
]
