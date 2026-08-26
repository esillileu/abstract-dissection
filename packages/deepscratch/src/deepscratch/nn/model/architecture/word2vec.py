"""Word2Vec prediction architectures and explicit batch adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from deepscratch.core import Tensor
from deepscratch.core.backend import resolve_backend
from deepscratch.nn.kernels import negative_sampling_loss_gradient
from deepscratch.nn.model.base import Model
from deepscratch.nn.types import Parameter

ScatterAdd = Callable[[object, object, object], None]


def _select_scatter_add(backend) -> ScatterAdd:
    """Select the row-scatter primitive once when a model is constructed."""
    if backend.is_gpu:
        from cupyx import scatter_add

        xp = backend.xp

        def cupy_scatter_add(destination, indices, values) -> None:
            values = xp.broadcast_to(
                values,
                (*indices.shape, values.shape[-1]),
            )
            scatter_add(
                destination,
                indices.reshape(-1),
                values.reshape(-1, values.shape[-1]),
            )

        return cupy_scatter_add

    xp = backend.xp

    def backend_add_at(destination, indices, values) -> None:
        """Portable CPU fallback using the backend's indexed add."""
        values = xp.broadcast_to(
            values,
            (*indices.shape, values.shape[-1]),
        )
        xp.add.at(
            destination,
            indices.reshape(-1),
            values.reshape(-1, values.shape[-1]),
        )

    return backend_add_at


class _EmbeddingArchitecture(Model):
    def __init__(self, vocab_size: int, embedding_size: int, *, backend=None) -> None:
        resolved = resolve_backend(backend)
        super().__init__(resolved)
        self._scatter_add = _select_scatter_add(resolved)
        rng = resolved.random_stream("model_init")
        self.W_in = Parameter(
            (0.01 * rng.randn(vocab_size, embedding_size)).astype(resolved.float_dtype),
            backend=resolved,
            name="W_in",
        )
        self.W_out = Parameter(
            (0.01 * rng.randn(vocab_size, embedding_size)).astype(resolved.float_dtype),
            backend=resolved,
            name="W_out",
        )
        self._cache = None
        self._sparse_rows = None

    def sparse_parameter_rows(self):
        if self._sparse_rows is None:
            raise RuntimeError(
                "negative-sampling backward must run before sparse update"
            )
        return self._sparse_rows

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
        candidate_weights = (
            None if candidate_ids is None else self.W_out.data[candidate_ids]
        )
        scores = (
            hidden @ self.W_out.data.T
            if candidate_weights is None
            else self._candidate_scores(hidden, candidate_weights)
        )
        if cache:
            self._cache = (
                source,
                hidden,
                candidate_ids,
                candidate_weights,
            )
        return Tensor(scores, backend=self.backend)

    def backward_manual(self, gradient: Tensor) -> None:
        if self._cache is None:
            raise RuntimeError("forward(cache=True) must be called before backward")
        source, hidden, candidates, candidate_weights = self._cache
        if candidates is None:
            self.W_out.grad[...] = gradient.data.T @ hidden
            hidden_gradient = gradient.data @ self.W_out.data
        else:
            if candidate_weights is None:
                raise RuntimeError("candidate weights are unavailable")
            self.W_out.grad[...] = 0
            hidden_view = hidden.reshape(
                (len(hidden),) + (1,) * (candidates.ndim - 1) + (hidden.shape[-1],)
            )
            contribution = gradient.data[..., None] * hidden_view
            if self.backend.device == "cpu":
                _accumulate_rows(
                    self.backend,
                    self.W_out.grad,
                    candidates,
                    contribution,
                )
            else:
                self._scatter_add(
                    self.W_out.grad,
                    candidates,
                    contribution,
                )
            hidden_gradient = self.backend.xp.matmul(
                gradient.data.reshape(len(hidden), 1, -1),
                candidate_weights.reshape(
                    len(hidden),
                    -1,
                    hidden.shape[-1],
                ),
            )[:, 0, :]
            self._sparse_rows = {
                "W_in": source.reshape(-1),
                "W_out": candidates.reshape(-1),
            }
        self._backward_embedding(source, hidden_gradient)

    def _candidate_scores(self, hidden, candidate_weights):
        if candidate_weights.ndim > 3:
            flattened = candidate_weights.reshape(
                len(hidden),
                -1,
                hidden.shape[-1],
            )
            return self.backend.xp.matmul(
                flattened,
                hidden[..., None],
            )[..., 0].reshape(candidate_weights.shape[:-1])
        hidden_view = hidden.reshape(
            (len(hidden),) + (1,) * (candidate_weights.ndim - 2) + (hidden.shape[-1],)
        )
        return self.backend.xp.sum(
            hidden_view * candidate_weights,
            axis=-1,
        )

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


@dataclass(frozen=True)
class CBOWBatchAdapter:
    def prepare(self, contexts: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        return contexts, targets.reshape(-1)


@dataclass(frozen=True)
class SkipGramBatchAdapter:
    def prepare(self, contexts: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        """Keep unique centers and group context labels for every objective."""
        return targets.reshape(-1), contexts


@dataclass(frozen=True)
class PairExpandedSkipGramBatchAdapter:
    """Expand each center-context pair into an independent prediction row."""

    def prepare(self, contexts: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        if contexts.ndim != 2 or targets.ndim != 1 or len(contexts) != len(targets):
            raise ValueError("Skip-gram pair expansion expects (B, C) and (B,) batches")
        xp = contexts.backend.xp
        context_count = contexts.shape[1]
        centers = xp.repeat(targets.data.reshape(-1, 1), context_count, axis=1).reshape(
            -1
        )
        return (
            Tensor(centers, backend=contexts.backend),
            contexts.reshape(-1),
        )


@dataclass(frozen=True)
class OneHotCBOWBatchAdapter:
    vocab_size: int

    def prepare(self, contexts: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        return (
            _one_hot_tensor(contexts, self.vocab_size),
            _one_hot_tensor(targets.reshape(-1), self.vocab_size),
        )


@dataclass(frozen=True)
class OneHotSkipGramBatchAdapter:
    vocab_size: int

    def prepare(self, contexts: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        return (
            _one_hot_tensor(targets.reshape(-1), self.vocab_size),
            _one_hot_tensor(contexts, self.vocab_size),
        )


def _one_hot_tensor(values: Tensor, vocab_size: int) -> Tensor:
    xp = values.backend.xp
    indices = values.data.astype(xp.int64, copy=False)
    return Tensor(
        _one_hot(xp, indices, vocab_size, values.backend.float_dtype),
        backend=values.backend,
    )


def _one_hot(xp, indices, vocab_size: int, dtype):
    values = xp.zeros((*indices.shape, vocab_size), dtype=dtype)
    flat_values = values.reshape(-1, vocab_size)
    flat_indices = indices.reshape(-1)
    flat_values[xp.arange(len(flat_indices)), flat_indices] = 1
    return values


def _accumulate_rows(backend, destination, indices, values) -> None:
    """Accumulate indexed rows, coalescing CPU duplicates before assignment."""
    xp = backend.xp
    flat_indices = indices.reshape(-1)
    flat_values = values.reshape(-1, values.shape[-1])
    if backend.device != "cpu":
        xp.add.at(destination, flat_indices, flat_values)
        return

    order = xp.argsort(flat_indices)
    sorted_indices = flat_indices[order]
    sorted_values = flat_values[order]
    starts = xp.concatenate(
        (
            xp.asarray([0], dtype=xp.int64),
            xp.flatnonzero(sorted_indices[1:] != sorted_indices[:-1]) + 1,
        )
    )
    destination[sorted_indices[starts]] = xp.add.reduceat(
        sorted_values,
        starts,
        axis=0,
    )
