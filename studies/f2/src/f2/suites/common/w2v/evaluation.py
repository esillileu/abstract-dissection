"""Deterministic evaluation policies and compatibility facades for Word2Vec lookup artifacts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .analogy import (
    _analogy_rows,
    evaluate_analogies,
    parse_analogy_questions,
)
from .evaluation_types import (
    AnalogyEvaluation,
    AnalogyQuestion,
    EvaluationResult,
    NearestNeighbor,
    SimilarityPair,
)
from .geometry import (
    VectorLookup,
    additive_composition,
    nearest_tokens,
    pca_projection,
)
from .msr_completion import (
    SentenceCompletionEvaluation,
    SentenceCompletionQuestion,
    evaluate_msr_sentence_completion,
    parse_msr_sentence_completion_questions,
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
    score = _pearson(_ranks(human), _ranks(predicted)) if len(human) >= 2 else None
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
    """Choose the input candidate with greatest summed output-context score."""
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
        correct / valid if valid else None,
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
    return max(
        results,
        key=lambda item: (
            (item[1].require_score() if maximize else -item[1].require_score()),
            item[1].coverage,
            -item[0],
        ),
    )


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


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    left_centered = left_values - left_values.mean()
    right_centered = right_values - right_values.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    return (
        float(np.dot(left_centered, right_centered) / denominator)
        if denominator
        else None
    )


__all__ = [
    "AnalogyEvaluation",
    "AnalogyQuestion",
    "EvaluationResult",
    "NearestNeighbor",
    "SentenceCompletionEvaluation",
    "SentenceCompletionQuestion",
    "SimilarityPair",
    "VectorLookup",
    "_analogy_rows",
    "additive_composition",
    "evaluate_analogies",
    "evaluate_msr_sentence_completion",
    "evaluate_sentence_completion",
    "evaluate_word_similarity",
    "nearest_tokens",
    "parse_analogy_questions",
    "parse_msr_sentence_completion_questions",
    "pca_projection",
    "select_best_epoch",
]
