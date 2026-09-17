"""DS2 GT02: inspect final PTB Word2Vec embeddings with the book queries."""

from __future__ import annotations

# Re-export patchable collaborators so tests can monkeypatch on this module.
from deepscratch.datasets import load_ptb

from dlfs.analysis.input import artifact_file
from dlfs.ds2.analysis.common import source_curve

from .constants import (
    ANALOGY_QUERIES,
    ATOMIC_RUN_IDS,
    CSV_FIELDS,
    CURVE_ATOMIC_RUN_IDS,
    ORIGINAL_NATIVE_IDS,
    SIMILARITY_QUERIES,
    TOP_K,
)
from .curves import (
    _curve_output_paths,
    _render_ns_curves,
)
from .evaluation import (
    AnalogyResult,
    RankedCandidate,
    RunEvaluation,
    SimilarityResult,
    _analogy,
    _checkpoint_weights_path,
    _nearest_words,
    _ordered_indices,
    _word_vectors,
    evaluate_vectors,
)
from .renderer import render
from .reporting import (
    _candidate_text,
    _csv_rows,
    _markdown_tables,
    _output_paths,
    _series_label,
    _text,
    _write_csv,
    append_markdown_report,
)

__all__ = [
    "ANALOGY_QUERIES",
    "ATOMIC_RUN_IDS",
    "CSV_FIELDS",
    "CURVE_ATOMIC_RUN_IDS",
    "ORIGINAL_NATIVE_IDS",
    "SIMILARITY_QUERIES",
    "TOP_K",
    "AnalogyResult",
    "RankedCandidate",
    "RunEvaluation",
    "SimilarityResult",
    "_analogy",
    "_candidate_text",
    "_checkpoint_weights_path",
    "_csv_rows",
    "_curve_output_paths",
    "_markdown_tables",
    "_nearest_words",
    "_ordered_indices",
    "_output_paths",
    "_render_ns_curves",
    "_series_label",
    "_text",
    "_word_vectors",
    "_write_csv",
    "append_markdown_report",
    "artifact_file",
    "evaluate_vectors",
    "load_ptb",
    "render",
    "source_curve",
]
