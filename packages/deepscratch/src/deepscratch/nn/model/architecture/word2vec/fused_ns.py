"""Fused negative-sampling mixin and optimized CBOW/SkipGram variants."""

from __future__ import annotations

from deepscratch.core import Tensor
from deepscratch.nn.kernels import negative_sampling_loss_gradient

from .embedding import CBOW, SkipGram


class _FusedNegativeSamplingArchitecture:
    def forward_negative_sampling(
        self,
        inputs: Tensor,
        candidates: Tensor,
        labels: Tensor,
        *,
        divisor: int,
        cache: bool = True,
    ) -> Tensor:
        hidden, source = self._encode(inputs)
        xp = self.backend.xp
        candidate_ids = candidates.data.astype(xp.int64, copy=False)
        candidate_weights = self.W_out.data[candidate_ids]
        losses, score_gradient = negative_sampling_loss_gradient(
            hidden,
            candidate_weights,
            labels.data,
            divisor=divisor,
            backend=self.backend,
        )
        if cache:
            self._negative_sampling_cache = (
                source,
                hidden,
                candidate_ids,
                candidate_weights,
                score_gradient,
            )
        return Tensor(losses.sum(), backend=self.backend)

    def backward_negative_sampling(self) -> None:
        if self._negative_sampling_cache is None:
            raise RuntimeError(
                "forward_negative_sampling(cache=True) must run before backward"
            )
        source, hidden, candidates, candidate_weights, score_gradient = (
            self._negative_sampling_cache
        )
        hidden_view = hidden.reshape(
            (len(hidden),) + (1,) * (candidates.ndim - 1) + (hidden.shape[-1],)
        )
        self.W_out.grad[...] = 0
        self._scatter_add(
            self.W_out.grad,
            candidates.reshape(-1),
            (score_gradient[..., None] * hidden_view).reshape(-1, hidden.shape[-1]),
        )
        hidden_gradient = self.backend.xp.matmul(
            score_gradient.reshape(len(hidden), 1, -1),
            candidate_weights.reshape(
                len(hidden),
                -1,
                hidden.shape[-1],
            ),
        )[:, 0, :]
        self._backward_fused_embedding(source, hidden_gradient)

    def _backward_fused_embedding(self, source, hidden_gradient) -> None:
        raise NotImplementedError


class FusedNegativeSamplingCBOW(
    _FusedNegativeSamplingArchitecture,
    CBOW,
):
    """Standalone CBOW optimized exclusively for fused negative sampling."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        *,
        backend=None,
    ) -> None:
        super().__init__(vocab_size, embedding_size, backend=backend)
        self._negative_sampling_cache = None

    def _backward_fused_embedding(self, source, hidden_gradient) -> None:
        width = source.shape[1]
        self.W_in.grad[...] = 0
        self._scatter_add(
            self.W_in.grad,
            source,
            hidden_gradient[:, None, :] / width,
        )


class FusedNegativeSamplingSkipGram(
    _FusedNegativeSamplingArchitecture,
    SkipGram,
):
    """Standalone SkipGram optimized exclusively for fused negative sampling."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        *,
        backend=None,
    ) -> None:
        super().__init__(vocab_size, embedding_size, backend=backend)
        self._negative_sampling_cache = None

    def _backward_fused_embedding(self, source, hidden_gradient) -> None:
        self.W_in.grad[...] = 0
        self._scatter_add(
            self.W_in.grad,
            source,
            hidden_gradient,
        )
