from __future__ import annotations

from mlprosection import Tensor
from mlprosection.core.backend import resolve_backend
from mlprosection.nn.functional import (
    binary_cross_entropy_with_logits,
    softmax_cross_entropy,
)
from mlprosection.nn.sampling import UnigramSampler
from mlprosection.nn.types import Parameter

from .base import Objective, ObjectiveResult


class FullSoftmax(Objective):
    def __init__(
        self, *, reduction: str = "mean", backend=None,
    ) -> None:
        resolved = resolve_backend(backend)
        super().__init__(resolved)
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction
        self._cache = None

    def forward_manual(
        self, prediction: Tensor, target: Tensor, *, output_weight: Parameter,
        cache: bool = True, replay_context=None, example_count: int | None = None,
    ) -> ObjectiveResult:
        hidden = prediction.data
        computation = softmax_cross_entropy(
            Tensor(hidden @ output_weight.data.T, backend=prediction.backend),
            target.reshape(-1),
            reduction="sum",
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
            self._cache = (hidden, gradient, output_weight, prediction.backend)
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
        hidden, gradient, output_weight, backend = self._cache
        output_weight.grad[...] = gradient.T @ hidden
        return Tensor(gradient @ output_weight.data, backend=backend)


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
        self, prediction: Tensor, target: Tensor, *, output_weight: Parameter,
        cache: bool = True, replay_context=None, example_count: int | None = None,
    ) -> ObjectiveResult:
        xp = self.backend.xp
        labels = target.data.reshape(-1).astype(xp.int64, copy=False)
        negatives = (
            self.sampler.sample(labels, sample_size=self.negative_samples)
            if replay_context is None
            else replay_context
        )
        candidates = xp.concatenate((labels[:, None], negatives), axis=1)
        hidden = prediction.data
        scores = xp.sum(
            hidden[:, None, :] * output_weight.data[candidates], axis=2
        )
        binary_targets = xp.zeros_like(scores)
        binary_targets[:, 0] = 1
        computation = binary_cross_entropy_with_logits(
            Tensor(scores, backend=prediction.backend),
            Tensor(binary_targets, backend=target.backend),
            reduction="sum",
        )
        candidate_count = self.negative_samples + 1
        reporting_divisor = (
            len(labels) * candidate_count
            if self.reduction == "mean"
            else len(labels)
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
            self._cache = (
                hidden,
                candidates,
                score_gradient.data,
                output_weight,
                prediction.backend,
            )
        return ObjectiveResult(
            loss,
            len(labels),
            negatives.copy(),
            reporting_loss=(
                reporting_loss if example_count is not None else None
            ),
        )

    def backward_manual(self) -> Tensor:
        if self._cache is None:
            raise RuntimeError("forward(cache=True) must be called before backward")
        hidden, candidates, gradient, output_weight, backend = self._cache
        xp = backend.xp
        output_weight.grad[...] = 0
        xp.add.at(
            output_weight.grad,
            candidates,
            gradient[:, :, None] * hidden[:, None, :],
        )
        return Tensor(
            xp.sum(
                gradient[:, :, None] * output_weight.data[candidates], axis=1
            ),
            backend=backend,
        )
