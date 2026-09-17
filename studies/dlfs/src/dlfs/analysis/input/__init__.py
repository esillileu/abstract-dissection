"""Materialized, canonical input for DeepScratch study renderers.

Renderers consume this model instead of querying MLflow or knowing whether a
selected run came from the canonical or quarantined legacy namespace.

Internal implementation is split across focused sub-modules:
* ``_model``        - AnalysisRun dataclass
* ``_store``        - PreparedAnalysisStore disk-cached artifact store
* ``_study_input``  - StudyAnalysisInput canonical result coordinator
* ``_facade``       - Free functions for artifact loading and querying
"""

from __future__ import annotations

from ._facade import (
    artifact_file,
    artifact_rows,
    curve_from_artifact,
    histories_from_artifact,
    local_artifact_root,
    metric_histories,
)
from ._model import AnalysisRun
from ._store import PreparedAnalysisStore
from ._study_input import StudyAnalysisInput

__all__ = [
    "AnalysisRun",
    "PreparedAnalysisStore",
    "StudyAnalysisInput",
    "artifact_file",
    "artifact_rows",
    "curve_from_artifact",
    "histories_from_artifact",
    "local_artifact_root",
    "metric_histories",
]
