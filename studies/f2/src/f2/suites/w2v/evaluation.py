"""Deterministic evaluation policies for Word2Vec lookup artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

import numpy as np


class VectorLookup(Protocol):
    """Minimal byte-token interface shared by in-memory and mmap lookups."""

    embeddings: np.ndarray

    def row(self, token: bytes) -> int | None: ...

    def vector(self, token: bytes) -> np.ndarray | None: ...


@dataclass(frozen=True)
class EvaluationResult:
    """A score together with the denominator and vocabulary coverage that produced it."""

    metric_id: str
    score: float
    total_count: int
    valid_count: int
    excluded_count: int

    @property
    def coverage(self) -> float:
        return self.valid_count / self.total_count if self.total_count else 0.0


@dataclass(frozen=True)
class AnalogyQuestion:
    category: str
    a: bytes
    b: bytes
    c: bytes
    expected: bytes


@dataclass(frozen=True)
class AnalogyEvaluation:
    overall: EvaluationResult
    semantic: EvaluationResult
    syntactic: EvaluationResult
    categories: tuple[EvaluationResult, ...]


@dataclass(frozen=True)
class SimilarityPair:
    left: bytes
    right: bytes
    human_score: float


@dataclass(frozen=True)
class SentenceCompletionQuestion:
    context: tuple[bytes, ...]
    candidates: tuple[bytes, ...]
    expected: bytes


@dataclass(frozen=True)
class NearestNeighbor:
    token: bytes
    score: float


def parse_analogy_questions(lines: Iterable[bytes]) -> tuple[AnalogyQuestion, ...]:
    """Parse the upstream ``questions-words`` section-and-four-token format."""
    category: str | None = None
    questions: list[AnalogyQuestion] = []
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(b":"):
            name = line[1:].strip()
            if not name:
                raise ValueError(f"empty analogy category on line {line_number}")
            category = name.decode("ascii")
            continue
        fields = line.split()
        if category is None or len(fields) != 4:
            raise ValueError(f"invalid analogy question on line {line_number}")
        questions.append(AnalogyQuestion(category, *fields))
    return tuple(questions)


def evaluate_analogies(
    lookup: VectorLookup, questions: Sequence[AnalogyQuestion]
) -> AnalogyEvaluation:
    """Score 3CosAdd analogies, excluding any question containing an OOV token."""
    normalized = _normalized_embeddings(lookup.embeddings)
    category_counts: dict[str, list[int]] = {}
    for question in questions:
        counts = category_counts.setdefault(question.category, [0, 0, 0])
        counts[0] += 1
        rows = [lookup.row(token) for token in _analogy_tokens(question)]
        if any(row is None for row in rows):
            continue
        a, b, c, expected = (int(row) for row in rows)
        counts[1] += 1
        query = normalized[b] - normalized[a] + normalized[c]
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            continue
        scores = normalized @ (query / norm)
        scores[[a, b, c]] = -np.inf
        counts[2] += int(int(np.argmax(scores)) == expected)

    categories = tuple(
        _accuracy_result(name, total, valid, correct)
        for name, (total, valid, correct) in sorted(category_counts.items())
    )
    semantic = _combine_accuracy(
        "analogy_semantic_accuracy",
        (result for result in categories if not result.metric_id.startswith("gram")),
    )
    syntactic = _combine_accuracy(
        "analogy_syntactic_accuracy",
        (result for result in categories if result.metric_id.startswith("gram")),
    )
    return AnalogyEvaluation(
        overall=_combine_accuracy("analogy_accuracy", categories),
        semantic=semantic,
        syntactic=syntactic,
        categories=categories,
    )


def evaluate_word_similarity(
    lookup: VectorLookup, pairs: Sequence[SimilarityPair]
) -> EvaluationResult:
    """Compute Spearman rank correlation after excluding OOV pairs."""
    human: list[float] = []
    predicted: list[float] = []
    for pair in pairs:
        left = lookup.vector(pair.left)
        right = lookup.vector(pair.right)
        if left is None or right is None:
            continue
        human.append(pair.human_score)
        predicted.append(_cosine(left, right))
    score = _pearson(_ranks(human), _ranks(predicted)) if len(human) >= 2 else 0.0
    return EvaluationResult(
        "word_similarity_spearman",
        score,
        len(pairs),
        len(human),
        len(pairs) - len(human),
    )


def evaluate_sentence_completion(
    lookup: VectorLookup,
    output_embeddings: np.ndarray,
    questions: Sequence[SentenceCompletionQuestion],
) -> EvaluationResult:
    """Choose the input candidate with greatest summed output-context score.

    Questions with any missing context/candidate token or an absent expected candidate are
    excluded. Candidate order is the deterministic tie-breaker.
    """
    outputs = np.asarray(output_embeddings)
    if outputs.shape != lookup.embeddings.shape or not np.isfinite(outputs).all():
        raise ValueError("output embeddings must be finite and match input embeddings")
    correct = 0
    valid = 0
    for question in questions:
        context_rows = [lookup.row(token) for token in question.context]
        candidate_rows = [lookup.row(token) for token in question.candidates]
        if (
            not context_rows
            or not question.candidates
            or question.expected not in question.candidates
            or any(row is None for row in (*context_rows, *candidate_rows))
        ):
            continue
        scores = [
            sum(
                float(np.dot(lookup.embeddings[int(candidate)], outputs[int(context)]))
                for context in context_rows
            )
            for candidate in candidate_rows
        ]
        predicted = question.candidates[int(np.argmax(scores))]
        correct += int(predicted == question.expected)
        valid += 1
    return EvaluationResult(
        "sentence_completion_accuracy",
        correct / valid if valid else 0.0,
        len(questions),
        valid,
        len(questions) - valid,
    )


def select_best_epoch(
    results: Sequence[tuple[int, EvaluationResult]], *, maximize: bool = True
) -> tuple[int, EvaluationResult]:
    """Select by score, then coverage, then the earliest epoch."""
    if not results:
        raise ValueError("at least one epoch result is required")
    direction = 1.0 if maximize else -1.0
    return max(
        results,
        key=lambda item: (direction * item[1].score, item[1].coverage, -item[0]),
    )


def nearest_tokens(
    lookup: VectorLookup, token: bytes, *, limit: int = 10
) -> tuple[NearestNeighbor, ...]:
    """Return deterministic cosine neighbors for a word or phrase token."""
    row = lookup.row(token)
    if row is None:
        return ()
    if limit < 1:
        raise ValueError("neighbor limit must be positive")
    normalized = _normalized_embeddings(lookup.embeddings)
    scores = normalized @ normalized[row]
    scores[row] = -np.inf
    ranked = sorted(
        (index for index in range(len(scores)) if np.isfinite(scores[index])),
        key=lambda index: (-float(scores[index]), _token_at(lookup, index)),
    )[:limit]
    return tuple(
        NearestNeighbor(_token_at(lookup, index), float(scores[index]))
        for index in ranked
    )


def additive_composition(
    lookup: VectorLookup, tokens: Sequence[bytes], *, limit: int = 10
) -> tuple[NearestNeighbor, ...]:
    """Rank tokens nearest to the normalized sum of all supplied token vectors."""
    rows = [lookup.row(token) for token in tokens]
    if not tokens or any(row is None for row in rows):
        return ()
    normalized = _normalized_embeddings(lookup.embeddings)
    query = normalized[[int(row) for row in rows]].sum(axis=0)
    norm = float(np.linalg.norm(query))
    if norm == 0.0:
        return ()
    scores = normalized @ (query / norm)
    scores[[int(row) for row in rows]] = -np.inf
    ranked = sorted(
        (index for index in range(len(scores)) if np.isfinite(scores[index])),
        key=lambda index: (-float(scores[index]), _token_at(lookup, index)),
    )[:limit]
    return tuple(
        NearestNeighbor(_token_at(lookup, i), float(scores[i])) for i in ranked
    )


def pca_projection(
    lookup: VectorLookup, tokens: Sequence[bytes]
) -> dict[bytes, tuple[float, float]]:
    """Return a sign-stable two-dimensional PCA projection for qualitative reports."""
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


def _analogy_tokens(question: AnalogyQuestion) -> tuple[bytes, bytes, bytes, bytes]:
    return question.a, question.b, question.c, question.expected


def _accuracy_result(
    metric_id: str, total: int, valid: int, correct: int
) -> EvaluationResult:
    return EvaluationResult(
        metric_id, correct / valid if valid else 0.0, total, valid, total - valid
    )


def _combine_accuracy(
    metric_id: str, results: Iterable[EvaluationResult]
) -> EvaluationResult:
    selected = tuple(results)
    total = sum(result.total_count for result in selected)
    valid = sum(result.valid_count for result in selected)
    correct = sum(round(result.score * result.valid_count) for result in selected)
    return EvaluationResult(
        metric_id, correct / valid if valid else 0.0, total, valid, total - valid
    )


def _normalized_embeddings(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("embeddings must be a finite rank-2 array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms != 0.0)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    left_centered = left_values - left_values.mean()
    right_centered = right_values - right_values.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    return (
        float(np.dot(left_centered, right_centered) / denominator)
        if denominator
        else 0.0
    )


__all__ = [
    "AnalogyEvaluation",
    "AnalogyQuestion",
    "EvaluationResult",
    "NearestNeighbor",
    "SentenceCompletionQuestion",
    "SimilarityPair",
    "additive_composition",
    "evaluate_analogies",
    "evaluate_sentence_completion",
    "evaluate_word_similarity",
    "nearest_tokens",
    "parse_analogy_questions",
    "pca_projection",
    "select_best_epoch",
]
