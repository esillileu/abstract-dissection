"""Word and phrase analogy benchmarking routines."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np

from .evaluation_types import (
    AnalogyEvaluation,
    AnalogyQuestion,
    EvaluationResult,
)

EVALUATION_CHUNK_ROWS = 4096


class VectorLookup(Protocol):
    embeddings: np.ndarray

    def row(self, token: bytes) -> int | None: ...

    def vector(self, token: bytes) -> np.ndarray | None: ...


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
    lookup: VectorLookup,
    questions: Sequence[AnalogyQuestion],
    *,
    vocabulary_limit: int | None = None,
    batch_size: int = 256,
) -> AnalogyEvaluation:
    """Score analogies with the original ``compute-accuracy`` protocol."""
    if batch_size < 1:
        raise ValueError("analogy batch_size must be positive")
    row_count = len(lookup.embeddings)
    if vocabulary_limit is not None:
        if vocabulary_limit < 1:
            raise ValueError("analogy vocabulary_limit must be positive")
        row_count = min(row_count, vocabulary_limit)
    wanted = {
        token.upper() for question in questions for token in _analogy_tokens(question)
    }
    analogy_rows = _analogy_rows(lookup, row_count, wanted=wanted)
    category_counts: dict[str, list[int]] = {}
    valid: list[tuple[AnalogyQuestion, tuple[int, int, int, int]]] = []
    for question in questions:
        counts = category_counts.setdefault(question.category, [0, 0, 0])
        counts[0] += 1
        rows = [analogy_rows.get(token.upper()) for token in _analogy_tokens(question)]
        if any(row is None for row in rows):
            continue
        counts[1] += 1
        valid.append((question, tuple(int(row) for row in rows)))

    question_rows = [item[1] for item in valid]
    for (question, row), (prediction, positive) in zip(
        valid,
        _analogy_predictions(lookup.embeddings[:row_count], question_rows, batch_size),
        strict=True,
    ):
        category_counts[question.category][2] += int(positive and prediction == row[3])

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


def _analogy_predictions(
    embeddings: np.ndarray,
    question_rows: Sequence[tuple[int, int, int, int]],
    batch_size: int,
) -> Iterable[tuple[int, bool]]:
    if not question_rows:
        return
    batches: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for start in range(0, len(question_rows), batch_size):
        rows = np.asarray(question_rows[start : start + batch_size], dtype=np.int64)
        flat = rows[:, :3].ravel()
        selected_chunks = [
            _normalized_embeddings(
                embeddings[flat[i : i + EVALUATION_CHUNK_ROWS]], dtype=np.float32
            )
            for i in range(0, len(flat), EVALUATION_CHUNK_ROWS)
        ]
        selected = np.concatenate(selected_chunks, axis=0).reshape(
            len(rows), 3, embeddings.shape[1]
        )
        queries_t = np.ascontiguousarray(
            (selected[:, 1] - selected[:, 0] + selected[:, 2]).T
        )
        best = np.full(len(rows), -np.inf, dtype=np.float32)
        predictions = np.zeros(len(rows), dtype=np.int64)
        columns = np.arange(len(rows))
        batches.append((rows, queries_t, best, predictions, columns))

    for offset in range(0, len(embeddings), EVALUATION_CHUNK_ROWS):
        block = _normalized_embeddings(
            embeddings[offset : offset + EVALUATION_CHUNK_ROWS], dtype=np.float32
        )
        for rows, queries_t, best, predictions, columns in batches:
            scores = block @ queries_t
            for column in range(3):
                excluded = rows[:, column] - offset
                inside = (excluded >= 0) & (excluded < len(block))
                scores[excluded[inside], columns[inside]] = -np.inf
            local = np.argmax(scores, axis=0)
            values = scores[local, columns]
            better = values > best
            predictions[better] = offset + local[better]
            best[better] = values[better]

    for _, _, best, predictions, _ in batches:
        yield from (
            (int(row), bool(score > 0.0))
            for row, score in zip(predictions, best, strict=True)
        )


def _analogy_tokens(question: AnalogyQuestion) -> tuple[bytes, bytes, bytes, bytes]:
    return question.a, question.b, question.c, question.expected


def _accuracy_result(
    metric_id: str, total: int, valid: int, correct: int
) -> EvaluationResult:
    return EvaluationResult(
        metric_id, correct / valid if valid else None, total, valid, total - valid
    )


def _combine_accuracy(
    metric_id: str, results: Iterable[EvaluationResult]
) -> EvaluationResult:
    selected = tuple(results)
    total = sum(result.total_count for result in selected)
    valid = sum(result.valid_count for result in selected)
    correct = sum(
        round(result.require_score() * result.valid_count)
        for result in selected
        if result.valid_count
    )
    return EvaluationResult(
        metric_id, correct / valid if valid else None, total, valid, total - valid
    )


def _analogy_rows(
    lookup: VectorLookup, row_count: int, *, wanted: set[bytes]
) -> dict[bytes, int]:
    row_count = min(row_count, len(lookup.embeddings))
    token_bytes = getattr(lookup, "token_bytes", None)
    offsets = getattr(lookup, "token_offsets", None)
    if token_bytes is not None and offsets is not None:
        rows: dict[bytes, int] = {}
        for row in range(row_count):
            token = bytes(token_bytes[offsets[row] : offsets[row + 1]]).upper()
            if token in wanted:
                rows.setdefault(token, row)
        return rows

    tokens = getattr(lookup, "tokens", ())
    if tokens:
        return {
            bytes(token).upper(): row
            for row, token in enumerate(tokens[:row_count])
            if bytes(token).upper() in wanted
        }
    raise ValueError("analogy evaluation requires lookup token arrays")


def _normalized_embeddings(
    embeddings: np.ndarray, *, dtype: np.dtype = np.float64
) -> np.ndarray:
    values = np.asarray(embeddings, dtype=dtype)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("embeddings must be a finite rank-2 array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms != 0.0)


__all__ = [
    "AnalogyEvaluation",
    "AnalogyQuestion",
    "_analogy_rows",
    "evaluate_analogies",
    "parse_analogy_questions",
]
