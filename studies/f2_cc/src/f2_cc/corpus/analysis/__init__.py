"""Feasibility analysis engine executing DuckDB queries over exported provenance Parquet logs."""

from __future__ import annotations

from .analyzer import FeasibilityAnalyzer
from .models import (
    AuditConvergencePoint,
    AuditStoppingVerification,
    CrawlStratumYield,
    DedupScenarioYield,
    FeasibilityReportData,
)

__all__ = [
    "AuditConvergencePoint",
    "AuditStoppingVerification",
    "CrawlStratumYield",
    "DedupScenarioYield",
    "FeasibilityAnalyzer",
    "FeasibilityReportData",
]
