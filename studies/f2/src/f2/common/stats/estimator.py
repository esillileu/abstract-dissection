"""Two-Phase Stratified Difference Estimator and survey sample residual correction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StratumResidualEstimate:
    stratum_key: str
    phase1_count: int
    audit_count: int
    audit_scale: float
    total_residual_words: float
    weighted_residual_words: float


class TwoPhaseStratifiedDifferenceEstimator:
    """Computes difference estimator combining Phase-1 proxy yields with Phase-2 audited residuals."""

    @staticmethod
    def compute_stratum_scale(phase1_count: int, audit_count: int) -> float:
        """Compute phase-2 expansion scale factor N_1 / n_2."""
        return float(phase1_count) / float(max(1, audit_count))


__all__ = [
    "StratumResidualEstimate",
    "TwoPhaseStratifiedDifferenceEstimator",
]
