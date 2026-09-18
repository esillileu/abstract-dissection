"""Shared embedding base class, scatter primitive, and row-accumulation helper."""

from __future__ import annotations

from collections.abc import Callable

from deepscratch.core import Tensor
from deepscratch.core.backend import resolve_backend
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
