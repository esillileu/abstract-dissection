"""Shared, seed-aware MLflow analysis helpers for experiment domains.

Public API is identical to the previous single-file ``core.py``.  Internal
implementation is split across focused sub-modules:

* ``_client``   - AnalysisClient, RunRef, completed_seed_runs
* ``_cache``    - analysis_cache_path, cached_analysis_outputs, write_analysis_cache
* ``_artifacts``- artifact_file, artifact_rows, histories_from_artifact, metric_histories
* ``_smoothing``- smooth_book, smooth_histories
* ``_output``   - write_summary, parse_experiment_selection
"""

from __future__ import annotations

from ._artifacts import (
    artifact_file,
    artifact_rows,
    histories_from_artifact,
    metric_histories,
)
from ._cache import (
    ANALYSIS_CACHE_SCHEMA_VERSION,
    analysis_cache_path,
    cached_analysis_console_output,
    cached_analysis_outputs,
    write_analysis_cache,
)
from ._client import (
    AnalysisClient,
    RunRef,
    _local_artifact_root,
    completed_seed_runs,
)
from ._output import parse_experiment_selection, write_summary
from ._smoothing import smooth_book, smooth_histories

__all__ = [
    "ANALYSIS_CACHE_SCHEMA_VERSION",
    "AnalysisClient",
    "RunRef",
    "_local_artifact_root",
    "analysis_cache_path",
    "artifact_file",
    "artifact_rows",
    "cached_analysis_console_output",
    "cached_analysis_outputs",
    "completed_seed_runs",
    "histories_from_artifact",
    "metric_histories",
    "parse_experiment_selection",
    "smooth_book",
    "smooth_histories",
    "write_analysis_cache",
    "write_summary",
]
