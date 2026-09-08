"""Tests for f2.common.stats and f2.common.analysis."""

from __future__ import annotations

from pathlib import Path

from f2.common.analysis import (
    BaseAnalysisOrchestrator,
    ConditionDeclaration,
    MetricDeclaration,
    StudyDeclaration,
    apply_f2_plot_theme,
)
from f2.common.stats import (
    BootstrapVarianceEngine,
    ClassificationMetrics,
    TwoPhaseStratifiedDifferenceEstimator,
    binary_classification_metrics,
    bootstrap_confidence_interval,
    optimal_threshold_search,
)


def test_binary_classification_metrics():
    y_true = [1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 1, 1]
    metrics = binary_classification_metrics(y_true, y_pred)
    assert isinstance(metrics, ClassificationMetrics)
    assert metrics.tp == 2
    assert metrics.fn == 1
    assert metrics.fp == 1
    assert metrics.tn == 1
    assert metrics.precision == 2 / 3
    assert metrics.recall == 2 / 3


def test_optimal_threshold_search():
    y_true = [1, 1, 0, 0]
    y_scores = [0.9, 0.8, 0.3, 0.1]
    best_th, metrics = optimal_threshold_search(
        y_true, y_scores, thresholds=[0.2, 0.5, 0.85]
    )
    assert best_th == 0.5
    assert metrics.f1 == 1.0


def test_bootstrap_variance_engine():
    engine = BootstrapVarianceEngine(reps=100, seed=42)
    mults = engine.draw_gamma_multipliers(10)
    assert len(mults) == 10
    assert all(m >= 0.0 for m in mults)


def test_bootstrap_confidence_interval():
    estimates = [10.0, 20.0, 30.0, 40.0, 50.0]
    low, high = bootstrap_confidence_interval(estimates, alpha=0.1)
    assert low <= high


def test_two_phase_estimator_stratum_scale():
    scale = TwoPhaseStratifiedDifferenceEstimator.compute_stratum_scale(100, 10)
    assert scale == 10.0


def test_analysis_orchestrator(tmp_path: Path):
    orch = BaseAnalysisOrchestrator(tmp_path / "analysis_out")
    summary = orch.summarize_metric_series([1.0, 2.0, 3.0, 4.0, 5.0], "loss", "cbow")
    assert summary.mean_value == 3.0
    assert summary.sample_size == 5

    csv_path = orch.write_observations_csv(
        [{"condition": "cbow", "loss": 3.0}], "obs.csv"
    )
    assert csv_path.exists()
    assert "condition,loss" in csv_path.read_text(encoding="utf-8")


def test_declarations_and_theme():
    apply_f2_plot_theme()
    metric = MetricDeclaration("loss", "nll", "train", "loss", ("loss_val",))
    condition = ConditionDeclaration("cbow", ("cbow_alias",), (metric,))
    study = StudyDeclaration("w2v", (condition,))
    assert study.study_id == "w2v"
    assert study.conditions[0].canonical_id == "cbow"
