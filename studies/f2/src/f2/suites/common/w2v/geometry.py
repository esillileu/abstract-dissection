"""Geometric vector operations: nearest neighbors, additive composition, and PCA."""

from __future__ import annotations

from collections.abc import Sequence
from heapq import nsmallest
from typing import Protocol

import numpy as np

from .evaluation_types import NearestNeighbor

EVALUATION_CHUNK_ROWS = 16384


class VectorLookup(Protocol):
    embeddings: np.ndarray

    def row(self, token: bytes) -> int | None: ...


def nearest_tokens(
    lookup: VectorLookup, token: bytes, *, limit: int = 10
) -> tuple[NearestNeighbor, ...]:
    """Return deterministic cosine neighbors for a word or phrase token."""
    row = lookup.row(token)
    if row is None:
        return ()
    if limit < 1:
        raise ValueError("neighbor limit must be positive")
    query = _normalized_embeddings(lookup.embeddings[[row]])[0]
    if not np.any(query):
        return ()
    return _nearest_query(lookup, query, {row}, limit)


def _nearest_query(lookup, query, excluded, limit):
    def candidates():
        for offset in range(0, len(lookup.embeddings), EVALUATION_CHUNK_ROWS):
            block = _normalized_embeddings(
                lookup.embeddings[offset : offset + EVALUATION_CHUNK_ROWS]
            )
            scores = block @ query
            nonzero = np.any(block, axis=1)
            for local, score in enumerate(scores):
                row = offset + local
                if row not in excluded and np.isfinite(score) and nonzero[local]:
                    yield (-float(score), _token_at(lookup, row), row)

    ranked = nsmallest(limit, candidates())
    return tuple(NearestNeighbor(token, -score) for score, token, _ in ranked)


def additive_composition(
    lookup: VectorLookup, tokens: Sequence[bytes], *, limit: int = 10
) -> tuple[NearestNeighbor, ...]:
    """Rank tokens nearest to the normalized sum with bounded candidate memory."""
    if limit < 1:
        raise ValueError("neighbor limit must be positive")
    rows = [lookup.row(token) for token in tokens]
    if not tokens or any(row is None for row in rows):
        return ()
    rows = [int(row) for row in rows]
    inputs = _normalized_embeddings(lookup.embeddings[rows])
    if not np.all(np.any(inputs, axis=1)):
        return ()
    query = inputs.sum(axis=0)
    norm = float(np.linalg.norm(query))
    if norm == 0.0:
        return ()
    return _nearest_query(lookup, query / norm, set(rows), limit)


def pca_projection(
    lookup: VectorLookup, tokens: Sequence[bytes]
) -> dict[bytes, tuple[float, float]]:
    """Return a sign-stable two-dimensional PCA projection for qualitative reports."""
    if not tokens:
        return {}
    rows = [lookup.row(token) for token in tokens]
    if any(row is None for row in rows):
        return {}
    values = np.asarray(lookup.embeddings[[int(row) for row in rows]], dtype=np.float64)
    values -= values.mean(axis=0)
    _, _, right = np.linalg.svd(values, full_matrices=False)
    components = right[: min(2, len(right))].copy()
    for component in components:
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            component *= -1
    projected = values @ components.T
    if projected.shape[1] == 1:
        projected = np.column_stack((projected, np.zeros(len(projected))))
    return {
        token: (float(point[0]), float(point[1]))
        for token, point in zip(tokens, projected, strict=True)
    }


def _token_at(lookup: VectorLookup, row: int) -> bytes:
    token_bytes = getattr(lookup, "token_bytes", None)
    offsets = getattr(lookup, "token_offsets", None)
    if token_bytes is None or offsets is None:
        raise ValueError("neighbor evaluation requires lookup token arrays")
    return bytes(token_bytes[offsets[row] : offsets[row + 1]])


def _normalized_embeddings(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return np.asarray(matrix / norm, dtype=np.float32)


__all__ = [
    "EVALUATION_CHUNK_ROWS",
    "additive_composition",
    "nearest_tokens",
    "pca_projection",
]
