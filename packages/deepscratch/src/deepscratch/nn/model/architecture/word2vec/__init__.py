"""Word2Vec prediction architectures and explicit batch adapters."""

from __future__ import annotations

from ._base import (
    ScatterAdd,
    _accumulate_rows,
    _EmbeddingArchitecture,
    _select_scatter_add,
)
from .adapters import (
    CBOWBatchAdapter,
    OneHotCBOWBatchAdapter,
    OneHotSkipGramBatchAdapter,
    PairExpandedSkipGramBatchAdapter,
    SkipGramBatchAdapter,
    _one_hot,
    _one_hot_tensor,
)
from .embedding import CBOW, OneHotCBOW, OneHotSkipGram, SkipGram
from .full_softmax import DumbCBOW, DumbSkipGram, _DumbFullSoftmaxArchitecture
from .fused_ns import (
    FusedNegativeSamplingCBOW,
    FusedNegativeSamplingSkipGram,
    _FusedNegativeSamplingArchitecture,
)

__all__ = [
    "CBOW",
    "CBOWBatchAdapter",
    "DumbCBOW",
    "DumbSkipGram",
    "FusedNegativeSamplingCBOW",
    "FusedNegativeSamplingSkipGram",
    "OneHotCBOW",
    "OneHotCBOWBatchAdapter",
    "OneHotSkipGram",
    "OneHotSkipGramBatchAdapter",
    "PairExpandedSkipGramBatchAdapter",
    "ScatterAdd",
    "SkipGram",
    "SkipGramBatchAdapter",
    "_DumbFullSoftmaxArchitecture",
    "_EmbeddingArchitecture",
    "_FusedNegativeSamplingArchitecture",
    "_accumulate_rows",
    "_one_hot",
    "_one_hot_tensor",
    "_select_scatter_add",
]
