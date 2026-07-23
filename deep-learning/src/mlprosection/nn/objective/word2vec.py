from __future__ import annotations

from mlprosection import Tensor
from mlprosection.core.backend import resolve_backend
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
        xp = self.backend.xp
        labels = target.data.reshape(-1).astype(xp.int64, copy=False)
        hidden = prediction.data
        scores = hidden @ self.W_out.data.T
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = xp.exp(scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        terms = -xp.log(probabilities[xp.arange(len(labels)), labels] + 1e-7)
        value = terms.mean() if self.reduction == "mean" else terms.sum()
        if cache:
            self._cache = (hidden, labels, probabilities, prediction.backend)
        return ObjectiveResult(
            Tensor(xp.asarray(value, dtype=self.backend.float_dtype), backend=self.backend),
            len(labels),
        )

    def backward_manual(self) -> Tensor:
        if self._cache is None:
            raise RuntimeError("forward(cache=True) must be called before backward")
        hidden, labels, probabilities, backend = self._cache
        xp = backend.xp
        gradient = probabilities.copy()
        gradient[xp.arange(len(labels)), labels] -= 1
        if self.reduction == "mean":
            gradient /= len(labels)
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
        probabilities = 1 / (1 + xp.exp(-scores))
        terms = -(
            binary_targets * xp.log(probabilities + 1e-7)
            + (1 - binary_targets) * xp.log(1 - probabilities + 1e-7)
        )
        value = (
            terms.mean()
            if self.reduction == "mean"
            else terms.sum(axis=1).mean()
        )
        if cache:
            self._cache = (
                hidden, candidates, probabilities, binary_targets, prediction.backend
            )
        return ObjectiveResult(
            Tensor(xp.asarray(value, dtype=self.backend.float_dtype), backend=self.backend),
            len(labels),
            negatives.copy(),
        )

    def backward_manual(self) -> Tensor:
        if self._cache is None:
            raise RuntimeError("forward(cache=True) must be called before backward")
        hidden, candidates, probabilities, targets, backend = self._cache
        xp = backend.xp
        denominator = (
            probabilities.size
            if self.reduction == "mean"
            else probabilities.shape[0]
        )
        gradient = (probabilities - targets) / denominator
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
