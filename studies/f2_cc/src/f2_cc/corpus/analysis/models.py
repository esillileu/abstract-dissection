"""Data models for feasibility analysis and report generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CrawlStratumYield:
    crawl_id: str
    proxy_total_words: float
    residual_error_words: float
    true_total_words: float
    std_error_words: float
    ci_lower_95: float
    ci_upper_95: float
    sample_size: int
    retained_news_docs: int
    weighted_ppv: float | None = None
    weighted_tpr: float | None = None


@dataclass(frozen=True)
class AuditConvergencePoint:
    budget: int
    audited_docs: int
    true_total_words: float
    std_error_words: float
    ci_lower_95: float
    ci_upper_95: float
    relative_standard_error: float
    lower_vs_33b_ratio: float
    strata_yields: list[CrawlStratumYield]


@dataclass(frozen=True)
class DedupScenarioYield:
    name: str
    dedup_rate: float
    net_point_words: float
    net_ci_lower_95: float
    net_ci_upper_95: float
    point_margin_vs_33b: float
    lower_margin_vs_33b: float


@dataclass(frozen=True)
class AuditStoppingVerification:
    relative_standard_error: float
    rse_threshold_met: bool
    dedup50_lower_margin: float
    dedup50_margin_met: bool
    stratum0_fn_rate: float
    fn_stability_met: bool
    inter_wave_drift: float
    drift_stability_met: bool
    all_criteria_satisfied: bool


@dataclass(frozen=True)
class FeasibilityReportData:
    strata_yields: list[CrawlStratumYield]
    aggregated_true_words: float
    aggregated_std_error: float
    aggregated_ci_lower_95: float
    aggregated_ci_upper_95: float
    scenarios: dict[str, float]
    dedup_scenarios: list[DedupScenarioYield]
    has_audit: bool
    audit_sample_size: int
    precision_ppv: float | None
    recall_tpr: float | None
    good_turing_coverage: float
    chao1_richness: float
    feasibility_1b: bool
    feasibility_6b: bool
    feasibility_33b: bool
    baseline_10k_comparison: dict[str, Any] | None = None
    provenance_metadata: dict[str, Any] | None = None
    sequential_funnel: list[dict[str, Any]] | None = None
    marginal_filters: list[dict[str, Any]] | None = None
    convergence_points: list[AuditConvergencePoint] | None = None
    stopping_verification: AuditStoppingVerification | None = None
