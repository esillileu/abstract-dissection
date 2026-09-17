"""Estimate DS2 e02-e04 runtimes through the book's official CuPy path."""

from __future__ import annotations

from .cli import main
from .lm import (
    _eval_perplexity_iterations,
    estimate_lstm_rnnlm,
    estimate_rnnlm,
)
from .schema import (
    B2_SOURCE_ROOT,
    BOOK_ROOT,
    DEFAULT_OUTPUT,
    PTB_TRAIN,
    RuntimeEstimate,
    _add_import_path,
)
from .w2v import (
    _prepare_word2vec,
    estimate_word2vec,
)

__all__ = [
    "B2_SOURCE_ROOT",
    "BOOK_ROOT",
    "DEFAULT_OUTPUT",
    "PTB_TRAIN",
    "RuntimeEstimate",
    "_add_import_path",
    "_eval_perplexity_iterations",
    "_prepare_word2vec",
    "estimate_lstm_rnnlm",
    "estimate_rnnlm",
    "estimate_word2vec",
    "main",
]
