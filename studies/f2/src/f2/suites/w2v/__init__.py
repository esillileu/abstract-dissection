"""Shared Word2Vec suite policies."""

from f2.suites.common.w2v.corpus import (
    CorpusBinding,
    CorpusMaterializer,
    CorpusShard,
    MaterializedCorpus,
)
from f2.suites.common.w2v.evaluation import EvaluationResult
from f2.suites.common.w2v.observations import (
    DenseObservationWriter,
    ObservationRow,
    sparse_metric_rows,
)
from f2.suites.common.w2v.spec import Word2VecRunSpec

from .executor import Word2VecExecutor, Word2VecResult

__all__ = [
    "CorpusBinding",
    "CorpusMaterializer",
    "CorpusShard",
    "DenseObservationWriter",
    "EvaluationResult",
    "MaterializedCorpus",
    "ObservationRow",
    "Word2VecExecutor",
    "Word2VecResult",
    "Word2VecRunSpec",
    "sparse_metric_rows",
]
