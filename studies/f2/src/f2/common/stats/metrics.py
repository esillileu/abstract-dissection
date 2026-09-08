"""Statistical classifier evaluation metrics, ROC/PR curves, and threshold calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ClassificationMetrics:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    f05: float
    accuracy: float
    specificity: float


def binary_classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> ClassificationMetrics:
    """Calculate standard binary classification metrics."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt == 0 and yp == 0)

    prec = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / max(1, tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / max(1e-9, prec + rec) if (prec + rec) > 0 else 0.0
    f05 = (
        (1.25 * prec * rec) / max(1e-9, 0.25 * prec + rec)
        if (0.25 * prec + rec) > 0
        else 0.0
    )
    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    spec = tn / max(1, tn + fp) if (tn + fp) > 0 else 0.0

    return ClassificationMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=prec,
        recall=rec,
        f1=f1,
        f05=f05,
        accuracy=acc,
        specificity=spec,
    )


def optimal_threshold_search(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    thresholds: Sequence[float],
    target_metric: str = "f05",
) -> tuple[float, ClassificationMetrics]:
    """Search for optimal classification decision threshold maximizing target_metric."""
    best_thresh = thresholds[0]
    best_metrics = None
    best_score = -1.0

    for thresh in thresholds:
        y_pred = [1 if s >= thresh else 0 for s in y_scores]
        metrics = binary_classification_metrics(y_true, y_pred)
        score = getattr(metrics, target_metric, metrics.f1)
        if score > best_score:
            best_score = score
            best_thresh = thresh
            best_metrics = metrics

    assert best_metrics is not None
    return best_thresh, best_metrics


__all__ = [
    "ClassificationMetrics",
    "binary_classification_metrics",
    "optimal_threshold_search",
]
