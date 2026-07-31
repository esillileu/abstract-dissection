from __future__ import annotations

from dataclasses import dataclass

from mlprosection import Tensor
from mlprosection.core.backend import resolve_backend
from mlprosection.nn.functional import (
    LossComputation,
    binary_cross_entropy_with_logits,
    softmax_cross_entropy,
)
from mlprosection.nn.sampling import UnigramSampler

from .base import Objective, ObjectiveResult


@dataclass(frozen=True)
class Word2VecObjectiveBatch:
    target: Tensor
    candidates: Tensor | None = None
    replay_context: object = None


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
        self, prediction: Tensor, target: Tensor, *, cache: bool = True,
        replay_context=None, example_count: int | None = None,
    ) -> ObjectiveResult:
        computation = (
            _grouped_softmax_cross_entropy(prediction, target)
            if self.grouped_targets
            else softmax_cross_entropy(
                prediction,
                target.reshape(-1),
                reduction="sum",
            )
        )
        prediction_count = computation.unit_count
        reporting_divisor = prediction_count if self.reduction == "mean" else 1
        reporting_loss = computation.loss / reporting_divisor
        optimized_divisor = (
            example_count
            if example_count is not None
            else reporting_divisor
        )
        loss = computation.loss / optimized_divisor
        gradient = computation.gradient.data / optimized_divisor
        if cache:
            self._cache = Tensor(gradient, backend=prediction.backend)
        return ObjectiveResult(
            loss,
            prediction_count,
            reporting_loss=(
                reporting_loss if example_count is not None else None
            ),
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
        prepared = target if self.grouped_targets else target.reshape(-1)
        return Word2VecObjectiveBatch(target=prepared)


def _grouped_softmax_cross_entropy(
    logits: Tensor,
    target: Tensor,
) -> LossComputation:
    """Sum cross-entropy terms for multiple labels sharing each logits row."""
    if logits.ndim != 2:
        raise ValueError("grouped softmax expects rank-2 logits")
    if target.ndim != 2:
        raise ValueError("grouped softmax expects rank-2 targets")
    if len(logits) != len(target):
        raise ValueError("grouped targets must match the logits batch size")
    if target.shape[1] < 1:
        raise ValueError("grouped softmax expects at least one target per example")

    xp = logits.backend.xp
    scores = logits.data
    labels = target.data.astype(xp.int64, copy=False)
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
        unit_count=int(target.size),
    )


class NegativeSampling(Objective):
    def __init__(
        self,
        vocab_size: int,
        *,
        negative_samples: int = 5,
        reduction: str = "mean",
        sampler: UnigramSampler | None = None,
        backend=None,
    ) -> None:
        resolved = resolve_backend(backend)
        super().__init__(resolved)
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction
        self.negative_samples = negative_samples
        self.sampler = sampler or UnigramSampler.uniform(vocab_size, backend=resolved)
        if self.sampler.vocab_size != vocab_size:
            raise ValueError("sampler vocabulary does not match objective vocabulary")
        self._cache = None

    def forward_manual(
        self, prediction: Tensor, target: Tensor, *, cache: bool = True,
        replay_context=None, example_count: int | None = None,
    ) -> ObjectiveResult:
        computation = binary_cross_entropy_with_logits(
            prediction,
            target,
            reduction="sum",
        )
        prediction_count, candidate_count = prediction.shape
        reporting_divisor = (
            prediction_count * candidate_count
            if self.reduction == "mean"
            else prediction_count
        )
        reporting_loss = computation.loss / reporting_divisor
        optimized_divisor = (
            example_count
            if example_count is not None
            else reporting_divisor
        )
        loss = computation.loss / optimized_divisor
        score_gradient = computation.gradient / optimized_divisor
        if cache:
            self._cache = score_gradient
        return ObjectiveResult(
            loss,
            prediction_count,
            replay_context,
            reporting_loss=(
                reporting_loss if example_count is not None else None
            ),
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
        xp = self.backend.xp
        labels = target.data.reshape(-1).astype(xp.int64, copy=False)
        negatives = (
            self.sampler.sample(labels, sample_size=self.negative_samples)
            if replay_context is None
            else replay_context
        )
        candidates = xp.concatenate((labels[:, None], negatives), axis=1)
        binary_targets = xp.zeros(
            candidates.shape,
            dtype=self.backend.float_dtype,
        )
        binary_targets[:, 0] = 1
        return Word2VecObjectiveBatch(
            target=Tensor(binary_targets, backend=target.backend),
            candidates=Tensor(candidates, backend=target.backend),
            replay_context=negatives.copy(),
        )
