"""Full-softmax mixin and dumb (non-fused) CBOW/SkipGram variants."""

from __future__ import annotations

from deepscratch.core import Tensor

from ._base import _accumulate_rows
from .embedding import CBOW, SkipGram


class _DumbFullSoftmaxArchitecture:
    """Evaluate full-softmax output weights as two independent classic layers."""

    full_softmax_shards = 2

    def forward_manual(
        self,
        inputs: Tensor,
        *,
        candidates: Tensor | None = None,
        cache: bool = True,
    ) -> Tensor:
        if candidates is not None:
            return super().forward_manual(
                inputs,
                candidates=candidates,
                cache=cache,
            )
        hidden, source = self._encode(inputs)
        xp = self.backend.xp
        output_shards = xp.array_split(
            self.W_out.data,
            self.full_softmax_shards,
            axis=0,
        )
        scores = xp.concatenate(
            [hidden @ shard.T for shard in output_shards],
            axis=1,
        )
        if cache:
            self._cache = (source, hidden, None, output_shards)
        return Tensor(scores, backend=self.backend)

    def backward_manual(self, gradient: Tensor) -> None:
        if self._cache is None:
            raise RuntimeError("forward(cache=True) must be called before backward")
        source, hidden, candidates, output_shards = self._cache
        if candidates is not None:
            super().backward_manual(gradient)
            return
        xp = self.backend.xp
        gradient_shards = xp.array_split(
            gradient.data,
            self.full_softmax_shards,
            axis=1,
        )
        self.W_out.grad[...] = 0
        hidden_gradient = xp.zeros_like(hidden)
        start = 0
        for gradient_shard, output_shard in zip(
            gradient_shards,
            output_shards,
            strict=True,
        ):
            stop = start + len(output_shard)
            self.W_out.grad[start:stop] = gradient_shard.T @ hidden
            hidden_gradient += gradient_shard @ output_shard
            start = stop
        self._backward_embedding(source, hidden_gradient)


class DumbCBOW(_DumbFullSoftmaxArchitecture, CBOW):
    """Classic non-fused CBOW used for direct architecture comparisons."""


class DumbSkipGram(_DumbFullSoftmaxArchitecture, SkipGram):
    """Classic non-fused SkipGram whose contexts are independent pairs."""

    def _candidate_scores(self, hidden, candidate_weights):
        if candidate_weights.ndim != 4:
            return super()._candidate_scores(hidden, candidate_weights)
        xp = self.backend.xp
        return xp.stack(
            [
                xp.sum(hidden[:, None, :] * candidate_weights[:, index], axis=-1)
                for index in range(candidate_weights.shape[1])
            ],
            axis=1,
        )

    def backward_manual(self, gradient: Tensor) -> None:
        if self._cache is None:
            raise RuntimeError("forward(cache=True) must be called before backward")
        source, hidden, candidates, candidate_weights = self._cache
        if candidates is None or candidates.ndim != 3:
            super().backward_manual(gradient)
            return

        xp = self.backend.xp
        self.W_out.grad[...] = 0
        hidden_gradient = xp.zeros_like(hidden)
        for index in range(candidates.shape[1]):
            context_candidates = candidates[:, index]
            context_gradient = gradient.data[:, index]
            contribution = context_gradient[..., None] * hidden[:, None, :]
            if self.backend.device == "cpu":
                _accumulate_rows(
                    self.backend,
                    self.W_out.grad,
                    context_candidates,
                    contribution,
                )
            else:
                self._scatter_add(
                    self.W_out.grad,
                    context_candidates,
                    contribution,
                )
            hidden_gradient += xp.matmul(
                context_gradient[:, None, :],
                candidate_weights[:, index],
            )[:, 0, :]
        self._sparse_rows = {
            "W_in": source.reshape(-1),
            "W_out": candidates.reshape(-1),
        }
        self._backward_embedding(source, hidden_gradient)
