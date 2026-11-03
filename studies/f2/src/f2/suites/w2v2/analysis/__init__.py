"""W2V2 paper reproduction analysis package."""

from __future__ import annotations

from f2.suites.common.w2v.evaluation import (
    additive_composition,
    nearest_tokens,
)

from .cache import (
    EVALUATION_PROTOCOL,
    SEEDS,
    W2V2EvaluationCache,
    load_dataset_questions,
    status_from_items,
    summarize,
    targets,
)
from .cache import (
    load_dataset_questions as _dataset,
)
from .evaluator import (
    evaluate_single_seed,
    filter_inventory,
    index_records_by_key,
    paper_reference,
    questions_sha256,
    resolve_question_files,
)
from .loader import (
    EvidenceMissing,
    extract_timing,
    load_run_artifacts,
    metric_dict,
    sanitize_relative_path,
)
from .orchestrator import generate_report
from .qualitative import evaluate_qualitative_query, queries_for
from .reporting import write_analysis_deliverables

__all__ = [
    "EVALUATION_PROTOCOL",
    "SEEDS",
    "EvidenceMissing",
    "W2V2EvaluationCache",
    "_dataset",
    "additive_composition",
    "evaluate_qualitative_query",
    "evaluate_single_seed",
    "extract_timing",
    "filter_inventory",
    "generate_report",
    "index_records_by_key",
    "load_dataset_questions",
    "load_run_artifacts",
    "metric_dict",
    "nearest_tokens",
    "paper_reference",
    "queries_for",
    "questions_sha256",
    "resolve_question_files",
    "sanitize_relative_path",
    "status_from_items",
    "summarize",
    "targets",
    "write_analysis_deliverables",
]
