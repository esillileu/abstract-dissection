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
        self._source = None

    @property
    def word_vectors(self):
        return self.W_in.data


class CBOW(_EmbeddingArchitecture):
    def forward_manual(self, contexts: Tensor, *, cache: bool = True) -> Tensor:
        xp = self.backend.xp
        indices = contexts.data.astype(xp.int64, copy=False)
        if cache:
            self._source = indices
        return Tensor(self.W_in.data[indices].mean(axis=1), backend=self.backend)

    def backward_manual(self, gradient: Tensor) -> None:
        if self._source is None:
            raise RuntimeError("forward(cache=True) must be called before backward")
        self.W_in.grad[...] = 0
        width = self._source.shape[1]
        self.backend.xp.add.at(
            self.W_in.grad, self._source, gradient.data[:, None, :] / width
        )


class SkipGram(_EmbeddingArchitecture):
    def forward_manual(self, centers: Tensor, *, cache: bool = True) -> Tensor:
        xp = self.backend.xp
        indices = centers.data.reshape(-1).astype(xp.int64, copy=False)
        if cache:
            self._source = indices
        return Tensor(self.W_in.data[indices], backend=self.backend)

    def backward_manual(self, gradient: Tensor) -> None:
        if self._source is None:
            raise RuntimeError("forward(cache=True) must be called before backward")
        self.W_in.grad[...] = 0
        self.backend.xp.add.at(self.W_in.grad, self._source, gradient.data)


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
