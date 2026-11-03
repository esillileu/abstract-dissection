"""Shared analysis standards, graph styling, and summary generation."""

from .declarations import ConditionDeclaration, MetricDeclaration, StudyDeclaration
from .orchestrator import BaseAnalysisOrchestrator, NormalizedMetricSummary
from .tables import (
    BaseEvaluationCache,
    format_duration,
    format_duration_mean_sd,
    format_mean_sd,
    observed_training_seconds,
    write_run_records_csv,
    write_table_summary,
)
from .targets import (
    PaperTargetMetric,
    TargetComparisonResult,
    compare_all_targets,
    compare_to_target,
)
from .theme import W2V_COLORS, apply_f2_plot_theme

__all__ = [
    "W2V_COLORS",
    "BaseAnalysisOrchestrator",
    "BaseEvaluationCache",
    "ConditionDeclaration",
    "MetricDeclaration",
    "NormalizedMetricSummary",
    "PaperTargetMetric",
    "StudyDeclaration",
    "TargetComparisonResult",
    "apply_f2_plot_theme",
    "compare_all_targets",
    "compare_to_target",
    "format_duration",
    "format_duration_mean_sd",
    "format_mean_sd",
    "observed_training_seconds",
    "write_run_records_csv",
    "write_table_summary",
]
