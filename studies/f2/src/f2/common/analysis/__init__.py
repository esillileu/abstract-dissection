"""Shared analysis standards, graph styling, and summary generation."""

from .declarations import ConditionDeclaration, MetricDeclaration, StudyDeclaration
from .orchestrator import BaseAnalysisOrchestrator, NormalizedMetricSummary
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
    "ConditionDeclaration",
    "MetricDeclaration",
    "NormalizedMetricSummary",
    "PaperTargetMetric",
    "StudyDeclaration",
    "TargetComparisonResult",
    "apply_f2_plot_theme",
    "compare_all_targets",
    "compare_to_target",
]
