"""Stateless neural-network objective computations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mlprosection import Tensor

Reduction = Literal["mean", "sum"]


@dataclass(frozen=True)
class LossComputation:
    """A loss value and its gradient with respect to the input prediction."""

    loss: Tensor
    gradient: Tensor
    unit_count: int


def softmax_cross_entropy(
    logits: Tensor,
    target: Tensor,
    *,
    reduction: Reduction = "mean",
    ignore_label: int | None = None,
) -> LossComputation:
    """Compute cross entropy over the final logits axis.

    ``target`` may contain class indices or one-hot rows matching ``logits``.
    Leading dimensions are treated as independent prediction units.
    """
    _validate_reduction(reduction)
    if logits.ndim < 1:
        raise ValueError("softmax_cross_entropy expects at least one logits axis")

    xp = logits.backend.xp
    class_count = logits.shape[-1]
    scores = logits.data.reshape(-1, class_count)
    labels = target.data
    if target.shape == logits.shape:
        labels = labels.argmax(axis=-1)
    labels = labels.reshape(-1).astype(xp.int64, copy=False)
    if labels.size != scores.shape[0]:
        raise ValueError("target shape does not match the logits prediction units")

    if ignore_label is None:
        mask = xp.ones(labels.shape, dtype=xp.bool_)
        safe_labels = labels
        unit_count = int(labels.size)
    else:
        mask = labels != ignore_label
        safe_labels = xp.where(mask, labels, 0)
        mask_count = mask.sum()
        unit_count = (
            int(mask_count)
            if logits.backend.is_cpu
            else logits.backend.scalar_to_int(mask_count)
        )
    if unit_count == 0:
        raise ValueError("softmax_cross_entropy has no non-ignored targets")

    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = xp.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    rows = xp.arange(labels.size)
    terms = -xp.log(probabilities[rows, safe_labels] + 1e-7)
    value = (terms * mask).sum()
    if reduction == "mean":
        value /= unit_count

    gradient = probabilities
    gradient[rows, safe_labels] -= 1
    gradient *= mask[:, None]
    if reduction == "mean":
        gradient /= unit_count

    return LossComputation(
        loss=Tensor(
            xp.asarray(value, dtype=logits.backend.float_dtype),
            backend=logits.backend,
        ),
        gradient=Tensor(gradient.reshape(logits.shape), backend=logits.backend),
        unit_count=unit_count,
    )


def binary_cross_entropy_with_logits(
    logits: Tensor,
    target: Tensor,
    *,
    reduction: Reduction = "mean",
) -> LossComputation:
    """Compute numerically stable binary cross entropy from logits."""
    _validate_reduction(reduction)
    if target.shape != logits.shape:
        raise ValueError("target shape must match logits shape")

    xp = logits.backend.xp
    labels = target.data
    scores = logits.data
    terms = (
        xp.maximum(scores, 0)
        - scores * labels
        + xp.log1p(xp.exp(-xp.abs(scores)))
    )
    value = terms.sum()
    gradient = 1 / (1 + xp.exp(-scores)) - labels
    unit_count = int(target.size)
    if reduction == "mean":
        value /= unit_count
        gradient /= unit_count

    return LossComputation(
        loss=Tensor(
            xp.asarray(value, dtype=logits.backend.float_dtype),
            backend=logits.backend,
        ),
        gradient=Tensor(gradient, backend=logits.backend),
        unit_count=unit_count,
    )


def _validate_reduction(reduction: str) -> None:
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")
