"""Count-based distributional representations used by DS2 chapter 2."""

from __future__ import annotations

import numpy as np


def create_cooccurrence_matrix(
    corpus: np.ndarray, vocab_size: int, window_size: int
) -> np.ndarray:
    """Match the book's dense, symmetric context-count construction."""
    corpus = np.asarray(corpus, dtype=np.int64)
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    for index, word_id in enumerate(corpus):
        for offset in range(1, window_size + 1):
            left = index - offset
            right = index + offset
            if left >= 0:
                matrix[word_id, corpus[left]] += 1
            if right < len(corpus):
                matrix[word_id, corpus[right]] += 1
    return matrix


def positive_pmi(matrix: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Compute the book's dense PPMI matrix without its progress printing."""
    counts = np.asarray(matrix)
    total = float(counts.sum())
    marginals = counts.sum(axis=0, dtype=np.float64)
    denominator = marginals[:, None] * marginals[None, :]
    ratio = np.divide(
        counts.astype(np.float64) * total,
        denominator,
        out=np.zeros_like(counts, dtype=np.float64),
        where=denominator != 0,
    )
    values = np.log2(ratio + eps)
    return np.maximum(values, 0).astype(np.float32, copy=False)


def factorize_ppmi(
    matrix: np.ndarray,
    *,
    method: str,
    components: int,
    seed: int,
    n_iter: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return analysis vectors, singular values, and optional right factors."""
    if method == "ppmi":
        return matrix, np.empty(0, dtype=np.float32), None
    if method == "svd":
        u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
        return (
            u[:, :components].astype(np.float32, copy=False),
            singular_values.astype(np.float32, copy=False),
            None,
        )
    if method == "randomized_svd":
        from sklearn.utils.extmath import randomized_svd

        u, singular_values, vt = randomized_svd(
            matrix,
            n_components=components,
            n_iter=n_iter,
            random_state=seed,
        )
        return (
            u.astype(np.float32, copy=False),
            singular_values.astype(np.float32, copy=False),
            vt.astype(np.float32, copy=False),
        )
    raise ValueError(f"unknown count-based factorization: {method}")
