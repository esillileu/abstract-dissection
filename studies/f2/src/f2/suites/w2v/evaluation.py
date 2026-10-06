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
    lookup: VectorLookup,
    questions: Sequence[AnalogyQuestion],
    *,
    vocabulary_limit: int | None = None,
    batch_size: int = 256,
) -> AnalogyEvaluation:
    """Score analogies with the original ``compute-accuracy`` protocol.

    Vocabulary and question tokens are compared after ASCII upper-casing, vectors are
    normalized individually, and candidates use ``b - a + c`` while excluding the
    three input rows. Questions containing any out-of-vocabulary token are excluded.
    """
    if batch_size < 1:
        raise ValueError("analogy batch_size must be positive")
    row_count = len(lookup.embeddings)
    if vocabulary_limit is not None:
        if vocabulary_limit < 1:
            raise ValueError("analogy vocabulary_limit must be positive")
        row_count = min(row_count, vocabulary_limit)
    normalized = _normalized_embeddings(lookup.embeddings[:row_count], dtype=np.float32)
    analogy_rows = _analogy_rows(lookup, row_count)
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

    for start in range(0, len(valid), batch_size):
        batch = valid[start : start + batch_size]
        rows = np.asarray([item[1] for item in batch], dtype=np.int64)
        queries = (
            normalized[rows[:, 1]] - normalized[rows[:, 0]] + normalized[rows[:, 2]]
        )
        scores = normalized @ queries.T
        columns = np.arange(len(batch))
        scores[rows[:, 0], columns] = -np.inf
        scores[rows[:, 1], columns] = -np.inf
        scores[rows[:, 2], columns] = -np.inf
        predictions = np.argmax(scores, axis=0)
        positive = np.max(scores, axis=0) > 0.0
        for (question, row), prediction, has_positive in zip(
            batch, predictions, positive, strict=True
        ):
            category_counts[question.category][2] += int(
                has_positive and int(prediction) == row[3]
            )

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


@dataclass(frozen=True)
class SentenceCompletionEvaluation:
    overall: EvaluationResult
    dev: EvaluationResult
    test: EvaluationResult
    predictions: tuple[bytes, ...]


def parse_msr_sentence_completion_questions(
    questions_lines: Iterable[bytes | str],
    answers_lines: Iterable[bytes | str],
) -> tuple[SentenceCompletionQuestion, ...]:
    """Parse the canonical 1,040 questions and answers from MSR Sentence Completion Challenge.

    Expects 5,200 non-empty question lines (5 candidate sentences per problem) and
    1,040 gold answer lines. Candidate sentence formatting (e.g. leading <s> and
    trailing </s>) is stripped.
    """

    def _to_bytes(val: bytes | str) -> bytes:
        return val.encode("utf-8") if isinstance(val, str) else val

    q_lines = [
        _to_bytes(line).strip() for line in questions_lines if _to_bytes(line).strip()
    ]
    a_lines = [
        _to_bytes(line).strip() for line in answers_lines if _to_bytes(line).strip()
    ]

    if len(q_lines) != 5200:
        raise ValueError(
            f"canonical MSR Sentence Completion requires exactly 5,200 question lines (1,040 questions * 5); found {len(q_lines)}"
        )
    if len(a_lines) != 1040:
        raise ValueError(
            f"canonical MSR Sentence Completion requires exactly 1,040 answer lines; found {len(a_lines)}"
        )

    parsed: list[SentenceCompletionQuestion] = []
    for i in range(1040):
        cand_sentences = [line.split() for line in q_lines[i * 5 : (i + 1) * 5]]
        ans_tokens = a_lines[i].split()

        # Strip sentence boundary tokens <s> and </s> if present
        cand_sentences = [
            tokens[1:-1]
            if len(tokens) >= 2 and tokens[0] == b"<s>" and tokens[-1] == b"</s>"
            else tokens
            for tokens in cand_sentences
        ]
        if (
            len(ans_tokens) >= 2
            and ans_tokens[0] == b"<s>"
            and ans_tokens[-1] == b"</s>"
        ):
            ans_tokens = ans_tokens[1:-1]

        target_len = len(cand_sentences[0])
        if any(len(s) != target_len for s in cand_sentences):
            raise ValueError(
                f"question {i + 1} candidates have mismatched lengths: {[len(s) for s in cand_sentences]}"
            )

        diff_positions = [
            pos
            for pos in range(target_len)
            if len({cand[pos] for cand in cand_sentences}) > 1
        ]
        if len(diff_positions) != 1:
            raise ValueError(
                f"question {i + 1} has {len(diff_positions)} differing token positions; expected exactly 1"
            )

        blank_pos = diff_positions[0]
        candidates = tuple(cand[blank_pos] for cand in cand_sentences)
        expected = ans_tokens[blank_pos]
        if expected not in candidates:
            raise ValueError(
                f"question {i + 1} expected word {expected!r} not in candidate choices {candidates!r}"
            )

        context = tuple(
            token for idx, token in enumerate(cand_sentences[0]) if idx != blank_pos
        )
        parsed.append(
            SentenceCompletionQuestion(
                context=context, candidates=candidates, expected=expected
            )
        )

    return tuple(parsed)


def evaluate_msr_sentence_completion(
    lookup: VectorLookup,
    questions: Sequence[SentenceCompletionQuestion],
    *,
    scoring: str = "cosine",
    output_embeddings: np.ndarray | None = None,
) -> SentenceCompletionEvaluation:
    """Evaluate 1,040 MSR Sentence Completion questions using Skip-gram prediction.

    Per Mikolov et al. (2013) §4.5:
    'we compute score of each sentence in the test set by using the unknown word
    at the input, and predict all surrounding words in a sentence. The final
    sentence score is then the sum of these individual predictions. Using the
    sentence scores, we choose the most likely sentence.'

    Scoring modes:
      - 'cosine' (default): Word vectors are L2-normalized. For each candidate c,
        score(c) = sum_{w in context, w in V} cos(v_c, v_w) = v_c_hat · (sum_{w in context} v_w_hat).
      - 'dot': Unnormalized dot product sum_{w in context} v_c · v_w.
      - 'output_dot': Candidate input embedding dot product with context output embeddings
        sum_{w in context} v_c · v'_w.
    """
    if scoring not in {"cosine", "dot", "output_dot"}:
        raise ValueError(f"unsupported sentence completion scoring mode: {scoring}")

    if scoring == "output_dot":
        if output_embeddings is None:
            raise ValueError("output_embeddings required for 'output_dot' scoring")
        outputs = np.asarray(output_embeddings, dtype=np.float32)
        if outputs.shape != lookup.embeddings.shape:
            raise ValueError("output_embeddings must match lookup embeddings shape")
    else:
        outputs = None

    if scoring == "cosine":
        norm = np.linalg.norm(lookup.embeddings, axis=1, keepdims=True)
        norm = np.where(norm == 0.0, 1.0, norm)
        input_vectors = (lookup.embeddings / norm).astype(np.float32)
    else:
        input_vectors = np.asarray(lookup.embeddings, dtype=np.float32)

    def _resolve_row(token: bytes) -> int | None:
        row = lookup.row(token)
        if row is not None:
            return row
        lower = token.lower()
        if lower != token and (r := lookup.row(lower)) is not None:
            return r
        cap = token.capitalize()
        if cap != token and (r := lookup.row(cap)) is not None:
            return r
        upper = token.upper()
        if upper != token and (r := lookup.row(upper)) is not None:
            return r
        return None

    correct_total = 0
    correct_dev = 0
    correct_test = 0
    predictions: list[bytes] = []

    dev_count = min(520, len(questions) // 2)
    test_count = len(questions) - dev_count

    for idx, question in enumerate(questions):
        if outputs is not None:
            ctx_vectors = [
                outputs[r]
                for token in question.context
                if (r := _resolve_row(token)) is not None
            ]
        else:
            ctx_vectors = [
                input_vectors[r]
                for token in question.context
                if (r := _resolve_row(token)) is not None
            ]

        context_sum = np.sum(ctx_vectors, axis=0) if ctx_vectors else None

        candidate_scores: list[float] = []
        for cand in question.candidates:
            cand_row = _resolve_row(cand)
            if cand_row is None:
                candidate_scores.append(-float("inf"))
            elif context_sum is None:
                candidate_scores.append(0.0)
            else:
                candidate_scores.append(
                    float(np.dot(input_vectors[cand_row], context_sum))
                )

        pred_idx = int(np.argmax(candidate_scores))
        predicted = question.candidates[pred_idx]
        predictions.append(predicted)

        is_correct = predicted == question.expected
        if is_correct:
            correct_total += 1
            if idx < dev_count:
                correct_dev += 1
            else:
                correct_test += 1

    overall_res = EvaluationResult(
        metric_id="msr_sentence_completion_accuracy",
        score=correct_total / len(questions) if questions else 0.0,
        total_count=len(questions),
        valid_count=len(questions),
        excluded_count=0,
    )
    dev_res = EvaluationResult(
        metric_id="msr_sentence_completion_dev_accuracy",
        score=correct_dev / dev_count if dev_count else 0.0,
        total_count=dev_count,
        valid_count=dev_count,
        excluded_count=0,
    )
    test_res = EvaluationResult(
        metric_id="msr_sentence_completion_test_accuracy",
        score=correct_test / test_count if test_count else 0.0,
        total_count=test_count,
        valid_count=test_count,
        excluded_count=0,
    )
    return SentenceCompletionEvaluation(
        overall=overall_res,
        dev=dev_res,
        test=test_res,
        predictions=tuple(predictions),
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


def _analogy_rows(lookup: VectorLookup, row_count: int) -> dict[bytes, int]:
    token_bytes = getattr(lookup, "token_bytes", None)
    offsets = getattr(lookup, "token_offsets", None)
    if token_bytes is not None and offsets is not None:
        rows: dict[bytes, int] = {}
        for row in range(row_count):
            token = bytes(token_bytes[offsets[row] : offsets[row + 1]]).upper()
            rows.setdefault(token, row)
        return rows

    # Lightweight in-memory test lookups do not expose packed token arrays.
    tokens = getattr(lookup, "tokens", ())
    if tokens:
        return {
            bytes(token).upper(): row for row, token in enumerate(tokens[:row_count])
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
    "SentenceCompletionEvaluation",
    "SentenceCompletionQuestion",
    "SimilarityPair",
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
