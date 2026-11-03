"""W2V1 table analysis package."""

from __future__ import annotations

from .table02 import (
    CORPUS_SOURCES,
    Table2EvaluationCache,
    Table2RunResult,
    _complete_conditions,
    analyze_table2,
    analyze_table2_sources,
    observed_training_seconds,
)

__all__ = [
    "CORPUS_SOURCES",
    "Table2EvaluationCache",
    "Table2RunResult",
    "_complete_conditions",
    "analyze_table2",
    "analyze_table2_sources",
    "observed_training_seconds",
]
