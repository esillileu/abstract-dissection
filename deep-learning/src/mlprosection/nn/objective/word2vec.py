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
        self, vocab_size: int, embedding_size: int, *, reduction: str = "mean",
        backend=None,
    ) -> None:
        resolved = resolve_backend(backend)
        super().__init__(resolved)
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction
        self.W_out = Parameter(
            (0.01 * resolved.xp.random.randn(vocab_size, embedding_size)).astype(
                resolved.float_dtype
            ),
            backend=resolved,
            name="W_out",
        )
        self._cache = None

    def forward_manual(
        self, prediction: Tensor, target: Tensor, *, cache: bool = True,
        replay_context=None,
    ) -> ObjectiveResult:
        hidden = prediction.data
        computation = softmax_cross_entropy(
            Tensor(hidden @ self.W_out.data.T, backend=prediction.backend),
            target.reshape(-1),
            reduction=self.reduction,
        )
        if cache:
            self._cache = (hidden, computation.gradient.data, prediction.backend)
        return ObjectiveResult(
            computation.loss,
            computation.unit_count,
        )

    def backward_manual(self) -> Tensor:
        if self._cache is None:
            raise RuntimeError("forward(cache=True) must be called before backward")
        hidden, gradient, backend = self._cache
        self.W_out.grad[...] = gradient.T @ hidden
        return Tensor(gradient @ self.W_out.data, backend=backend)


class NegativeSampling(Objective):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
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
        self.W_out = Parameter(
            (0.01 * resolved.xp.random.randn(vocab_size, embedding_size)).astype(
                resolved.float_dtype
            ),
            backend=resolved,
            name="W_out",
        )
        self._cache = None

    def forward_manual(
        self, prediction: Tensor, target: Tensor, *, cache: bool = True,
        replay_context=None,
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
            hidden[:, None, :] * self.W_out.data[candidates], axis=2
        )
        binary_targets = xp.zeros_like(scores)
        binary_targets[:, 0] = 1
        computation = binary_cross_entropy_with_logits(
            Tensor(scores, backend=prediction.backend),
            Tensor(binary_targets, backend=target.backend),
            reduction="mean" if self.reduction == "mean" else "sum",
        )
        loss = computation.loss
        score_gradient = computation.gradient
        if self.reduction == "sum":
            loss = loss / len(labels)
            score_gradient = score_gradient / len(labels)
        if cache:
            self._cache = (hidden, candidates, score_gradient.data, prediction.backend)
        return ObjectiveResult(
            loss,
            len(labels),
            negatives.copy(),
        )

    def backward_manual(self) -> Tensor:
        if self._cache is None:
            raise RuntimeError("forward(cache=True) must be called before backward")
        hidden, candidates, gradient, backend = self._cache
        xp = backend.xp
        self.W_out.grad[...] = 0
        xp.add.at(
            self.W_out.grad,
            candidates,
            gradient[:, :, None] * hidden[:, None, :],
        )
        return Tensor(
            xp.sum(
                gradient[:, :, None] * self.W_out.data[candidates], axis=1
            ),
            backend=backend,
        )
