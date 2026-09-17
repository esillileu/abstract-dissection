"""Numerical lockstep validation for the e02 fused Word2Vec paths."""

from __future__ import annotations

from .__main__ import main
from .metrics import (
    GRADIENT_CEILING,
    LOSS_CEILING,
    PARAMETER_CEILING,
    _error_metrics,
    _max_error,
    _optimizer_errors,
    _parameter_errors,
    _passed,
)
from .report import render_report
from .runner import (
    DEFAULT_OUTPUT,
    DEFAULT_REPORT,
    _batches,
    _copy_parameters,
    _tensor,
    run,
    validate_kind,
)

__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_REPORT",
    "GRADIENT_CEILING",
    "LOSS_CEILING",
    "PARAMETER_CEILING",
    "_batches",
    "_copy_parameters",
    "_error_metrics",
    "_max_error",
    "_optimizer_errors",
    "_parameter_errors",
    "_passed",
    "_tensor",
    "main",
    "render_report",
    "run",
    "validate_kind",
]
