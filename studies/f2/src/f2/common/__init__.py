"""Shared F2 composition, statistics and analysis standards."""

from .adapters import CheckpointAdapter
from .analysis import (
    W2V_COLORS,
    BaseAnalysisOrchestrator,
    ConditionDeclaration,
    MetricDeclaration,
    NormalizedMetricSummary,
    StudyDeclaration,
    apply_f2_plot_theme,
)
from .paths import (
    get_benchmark_data_dir,
    get_corpus_data_dir,
    get_f2_analysis_dir,
    get_f2_cache_dir,
    get_f2_paths,
    get_f2_staging_dir,
)
from .stats import (
    BootstrapVarianceEngine,
    ClassificationMetrics,
    StratumResidualEstimate,
    TwoPhaseStratifiedDifferenceEstimator,
    binary_classification_metrics,
    bootstrap_confidence_interval,
    optimal_threshold_search,
)

__all__ = [
    "W2V_COLORS",
    "BaseAnalysisOrchestrator",
    "BootstrapVarianceEngine",
    "CheckpointAdapter",
    "ClassificationMetrics",
    "ConditionDeclaration",
    "MetricDeclaration",
    "NormalizedMetricSummary",
    "StratumResidualEstimate",
    "StudyDeclaration",
    "TwoPhaseStratifiedDifferenceEstimator",
    "apply_f2_plot_theme",
    "binary_classification_metrics",
    "bootstrap_confidence_interval",
    "get_benchmark_data_dir",
    "get_corpus_data_dir",
    "get_f2_analysis_dir",
    "get_f2_cache_dir",
    "get_f2_paths",
    "get_f2_staging_dir",
    "optimal_threshold_search",
]
