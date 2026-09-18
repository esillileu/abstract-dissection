from __future__ import annotations

from deepscratch.core import Tensor
from deepscratch.core.backend import resolve_backend
from deepscratch.nn.functional import (
    LossComputation,
    softmax_cross_entropy,
)

from ..base import Objective, ObjectiveResult
from .batch import Word2VecObjectiveBatch


class SoftmaxWithLoss(Objective):
    def __init__(
        self,
        *,
        reduction: str = "mean",
        grouped_targets: bool = False,
        backend=None,
    ) -> None:
        resolved = resolve_backend(backend)
        super().__init__(resolved)
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction
        self.grouped_targets = grouped_targets
        self._cache = None

    def forward_manual(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        cache: bool = True,
        replay_context=None,
        example_count: int | None = None,
    ) -> ObjectiveResult:
        computation = (
            _grouped_softmax_cross_entropy(prediction, target)
            if self.grouped_targets
            else softmax_cross_entropy(
                prediction,
                target,
                reduction="sum",
            )
        )
        prediction_count = computation.unit_count
        reporting_divisor = prediction_count if self.reduction == "mean" else 1
        reporting_loss = computation.loss / reporting_divisor
        optimized_divisor = (
            example_count if example_count is not None else reporting_divisor
        )
        loss = computation.loss / optimized_divisor
        gradient = computation.gradient.data / optimized_divisor
        if cache:
            self._cache = Tensor(gradient, backend=prediction.backend)
        return ObjectiveResult(
            loss,
            prediction_count,
            reporting_loss=(reporting_loss if example_count is not None else None),
        )

    def backward_manual(self) -> Tensor:
        if self._cache is None:
            raise RuntimeError("forward(cache=True) must be called before backward")
        return self._cache

    def prepare(
        self,
        target: Tensor,
        *,
        replay_context=None,
    ) -> Word2VecObjectiveBatch:
        prepared = (
            target if self.grouped_targets or target.ndim > 1 else target.reshape(-1)
        )
        return Word2VecObjectiveBatch(target=prepared)


def _grouped_softmax_cross_entropy(
    logits: Tensor,
    target: Tensor,
) -> LossComputation:
    """Sum cross-entropy terms for multiple labels sharing each logits row."""
    if logits.ndim != 2:
        raise ValueError("grouped softmax expects rank-2 logits")
    if target.ndim not in {2, 3}:
        raise ValueError(
            "grouped softmax expects rank-2 labels or rank-3 one-hot targets"
        )
    if len(logits) != len(target):
        raise ValueError("grouped targets must match the logits batch size")
    if target.shape[1] < 1:
        raise ValueError("grouped softmax expects at least one target per example")

    xp = logits.backend.xp
    scores = logits.data
    if target.ndim == 3:
        if target.shape[-1] != logits.shape[-1]:
            raise ValueError("grouped one-hot target vocabulary does not match logits")
        labels = target.data.argmax(axis=-1)
    else:
        labels = target.data
    labels = labels.astype(xp.int64, copy=False)
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = xp.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    batch_rows = xp.arange(len(logits), dtype=xp.int64)
    grouped_rows = xp.broadcast_to(batch_rows[:, None], labels.shape)
    terms = -xp.log(probabilities[grouped_rows, labels] + 1e-7)
    value = terms.sum()

    gradient = probabilities * target.shape[1]
    xp.add.at(gradient, (grouped_rows, labels), -1)
    return LossComputation(
        loss=Tensor(
            xp.asarray(value, dtype=logits.backend.float_dtype),
            backend=logits.backend,
        ),
        gradient=Tensor(gradient, backend=logits.backend),
        unit_count=int(labels.size),
    )
