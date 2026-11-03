"""MSR sentence completion benchmark evaluation for Word2Vec embeddings."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .evaluation_types import EvaluationResult


class VectorLookup(Protocol):
    embeddings: np.ndarray

    def row(self, token: bytes) -> int | None: ...


@dataclass(frozen=True)
class SentenceCompletionQuestion:
    context: tuple[bytes, ...]
    candidates: tuple[bytes, ...]
    expected: bytes


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
    """Parse the canonical 1,040 questions and answers from MSR Sentence Completion Challenge."""

    def _to_bytes(val: bytes | str) -> bytes:
        return val.encode("utf-8") if isinstance(val, str) else val

    q_lines = [
        _to_bytes(line).strip() for line in questions_lines if _to_bytes(line).strip()
    ]
    a_lines = [
        _to_bytes(line).strip() for line in answers_lines if _to_bytes(line).strip()
    ]

    if len(q_lines) % 5 != 0 or len(q_lines) // 5 != len(a_lines):
        raise ValueError(
            f"canonical MSR Sentence Completion requires question lines to be 5x answer lines; found {len(q_lines)} questions and {len(a_lines)} answers"
        )

    num_questions = len(a_lines)
    parsed: list[SentenceCompletionQuestion] = []
    for i in range(num_questions):
        cand_sentences = [line.split() for line in q_lines[i * 5 : (i + 1) * 5]]
        ans_tokens = a_lines[i].split()

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
    """Evaluate 1,040 MSR Sentence Completion questions using Skip-gram prediction."""
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
        score=correct_total / len(questions) if questions else None,
        total_count=len(questions),
        valid_count=len(questions),
        excluded_count=0,
    )
    dev_res = EvaluationResult(
        metric_id="msr_sentence_completion_dev_accuracy",
        score=correct_dev / dev_count if dev_count else None,
        total_count=dev_count,
        valid_count=dev_count,
        excluded_count=0,
    )
    test_res = EvaluationResult(
        metric_id="msr_sentence_completion_test_accuracy",
        score=correct_test / test_count if test_count else None,
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


__all__ = [
    "SentenceCompletionEvaluation",
    "SentenceCompletionQuestion",
    "evaluate_msr_sentence_completion",
    "parse_msr_sentence_completion_questions",
]
