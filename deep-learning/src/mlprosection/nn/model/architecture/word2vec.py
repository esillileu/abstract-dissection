"""Word2Vec prediction architectures and explicit batch adapters."""

from __future__ import annotations

from dataclasses import dataclass

from mlprosection import Tensor
from mlprosection.core.backend import resolve_backend
from mlprosection.nn.model.base import Model
from mlprosection.nn.types import Parameter


class _EmbeddingArchitecture(Model):
    def __init__(self, vocab_size: int, embedding_size: int, *, backend=None) -> None:
        resolved = resolve_backend(backend)
        super().__init__(resolved)
        self.W_in = Parameter(
            (0.01 * resolved.xp.random.randn(vocab_size, embedding_size)).astype(
                resolved.float_dtype
            ),
            backend=resolved,
            name="W_in",
        )
        self.W_out = Parameter(
            (0.01 * resolved.xp.random.randn(vocab_size, embedding_size)).astype(
                resolved.float_dtype
            ),
            backend=resolved,
            name="W_out",
        )
        self._cache = None

    @property
    def word_vectors(self):
        return self.W_in.data

    def forward_manual(
        self,
        inputs: Tensor,
        *,
        candidates: Tensor | None = None,
        cache: bool = True,
    ) -> Tensor:
        hidden, source = self._encode(inputs)
        candidate_ids = (
            None
            if candidates is None
            else candidates.data.astype(self.backend.xp.int64, copy=False)
        )
        scores = (
            hidden @ self.W_out.data.T
            if candidate_ids is None
            else self.backend.xp.sum(
                hidden[:, None, :] * self.W_out.data[candidate_ids],
                axis=2,
            )
        )
        if cache:
            self._cache = (source, hidden, candidate_ids)
        return Tensor(scores, backend=self.backend)

    def backward_manual(self, gradient: Tensor) -> None:
        if self._cache is None:
            raise RuntimeError("forward(cache=True) must be called before backward")
        source, hidden, candidates = self._cache
        if candidates is None:
            self.W_out.grad[...] = gradient.data.T @ hidden
            hidden_gradient = gradient.data @ self.W_out.data
        else:
            self.W_out.grad[...] = 0
            self.backend.xp.add.at(
                self.W_out.grad,
                candidates,
                gradient.data[:, :, None] * hidden[:, None, :],
            )
            hidden_gradient = self.backend.xp.sum(
                gradient.data[:, :, None] * self.W_out.data[candidates],
                axis=1,
            )
        self._backward_embedding(source, hidden_gradient)

    def _encode(self, inputs: Tensor):
        raise NotImplementedError

    def _backward_embedding(self, source, gradient) -> None:
        raise NotImplementedError


class CBOW(_EmbeddingArchitecture):
    def _encode(self, contexts: Tensor):
        xp = self.backend.xp
        indices = contexts.data.astype(xp.int64, copy=False)
        return self.W_in.data[indices].mean(axis=1), indices

    def _backward_embedding(self, source, gradient) -> None:
        self.W_in.grad[...] = 0
        width = source.shape[1]
        self.backend.xp.add.at(
            self.W_in.grad,
            source,
            gradient[:, None, :] / width,
        )


class SkipGram(_EmbeddingArchitecture):
    def _encode(self, centers: Tensor):
        xp = self.backend.xp
        indices = centers.data.reshape(-1).astype(xp.int64, copy=False)
        return self.W_in.data[indices], indices

    def _backward_embedding(self, source, gradient) -> None:
        self.W_in.grad[...] = 0
        self.backend.xp.add.at(self.W_in.grad, source, gradient)


@dataclass(frozen=True)
class CBOWBatchAdapter:
    def prepare(self, contexts: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        return contexts, targets.reshape(-1)


@dataclass(frozen=True)
class SkipGramBatchAdapter:
    def prepare(self, contexts: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        """Vectorize all center→context prediction terms in one objective call."""
        xp = contexts.backend.xp
        centers = targets.data.reshape(-1)
        context_width = contexts.shape[1]
        return (
            Tensor(xp.repeat(centers, context_width), backend=targets.backend),
            Tensor(contexts.data.reshape(-1), backend=contexts.backend),
        )


@dataclass(frozen=True)
class SkipGramFullSoftmaxBatchAdapter:
    def prepare(self, contexts: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        """Keep unique centers and group their context labels for full softmax."""
        return targets.reshape(-1), contexts
