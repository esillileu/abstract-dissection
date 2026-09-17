"""Concrete embedding architectures: CBOW, SkipGram, OneHot variants."""

from __future__ import annotations

from deepscratch.core import Tensor

from ._base import _accumulate_rows, _EmbeddingArchitecture


class CBOW(_EmbeddingArchitecture):
    def _encode(self, contexts: Tensor):
        xp = self.backend.xp
        indices = contexts.data.astype(xp.int64, copy=False)
        return self.W_in.data[indices].mean(axis=1), indices

    def _backward_embedding(self, source, gradient) -> None:
        self.W_in.grad[...] = 0
        width = source.shape[1]
        if self.backend.device == "cpu":
            values = self.backend.xp.broadcast_to(
                gradient[:, None, :] / width,
                (*source.shape, gradient.shape[-1]),
            )
            _accumulate_rows(
                self.backend,
                self.W_in.grad,
                source,
                values,
            )
            return
        self._scatter_add(
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
        self._scatter_add(self.W_in.grad, source, gradient)


class OneHotCBOW(_EmbeddingArchitecture):
    """CBOW whose input projection explicitly multiplies one-hot contexts."""

    def _encode(self, contexts: Tensor):
        one_hot_contexts = contexts.data
        hidden = (one_hot_contexts @ self.W_in.data).mean(axis=1)
        return hidden, one_hot_contexts

    def _backward_embedding(self, source, gradient) -> None:
        width = source.shape[1]
        flattened_source = source.reshape(-1, source.shape[-1])
        flattened_gradient = self.backend.xp.repeat(
            (gradient / width)[:, None, :],
            width,
            axis=1,
        ).reshape(
            -1,
            gradient.shape[-1],
        )
        self.W_in.grad[...] = flattened_source.T @ flattened_gradient


class OneHotSkipGram(_EmbeddingArchitecture):
    """Skip-gram whose input projection explicitly multiplies one-hot centers."""

    def _encode(self, centers: Tensor):
        one_hot_centers = centers.data
        return one_hot_centers @ self.W_in.data, one_hot_centers

    def _backward_embedding(self, source, gradient) -> None:
        self.W_in.grad[...] = source.T @ gradient
