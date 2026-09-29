from __future__ import annotations

import numpy as np
import pytest

from f2.common.analysis.targets import PaperTargetMetric
from f2.suites.w2v.analysis import summarize_evaluations, summary_record
from f2.suites.w2v.evaluation import (
    EvaluationResult,
    SentenceCompletionQuestion,
    SimilarityPair,
    evaluate_analogies,
    evaluate_sentence_completion,
    evaluate_word_similarity,
    parse_analogy_questions,
    pca_projection,
    select_best_epoch,
)


class Lookup:
    def __init__(self, values: dict[bytes, tuple[float, ...]]) -> None:
        self.tokens = tuple(values)
        self.embeddings = np.asarray(tuple(values.values()), dtype=np.float32)
        self.rows = {token: row for row, token in enumerate(self.tokens)}

    def row(self, token: bytes) -> int | None:
        return self.rows.get(token)

    def vector(self, token: bytes) -> np.ndarray | None:
        row = self.row(token)
        return None if row is None else self.embeddings[row]


def test_empty_pca_projection_is_empty_without_numerical_warnings() -> None:
    lookup = Lookup({b"word": (1.0, 0.0)})
    assert pca_projection(lookup, []) == {}


def test_analogy_scoring_reports_oov_and_semantic_syntactic_splits() -> None:
    lookup = Lookup(
        {
            b"man": (1, 0),
            b"woman": (1, 1),
            b"king": (2, 0),
            b"queen": (2, 1),
            b"slow": (0, 1),
            b"slower": (-1, 1),
            b"fast": (0, 2),
            b"faster": (-1, 2),
        }
    )
    questions = parse_analogy_questions(
        [
            b": family\n",
            b"man woman king queen\n",
            b"missing woman king queen\n",
            b": gram3-comparative\n",
            b"slow slower fast faster\n",
        ]
    )
    result = evaluate_analogies(lookup, questions)

    assert result.overall.score == 1.0
    assert (result.overall.total_count, result.overall.valid_count) == (3, 2)
    assert result.overall.coverage == 2 / 3
    assert result.semantic.valid_count == 1
    assert result.syntactic.valid_count == 1
    assert [category.metric_id for category in result.categories] == [
        "family",
        "gram3-comparative",
    ]


def test_analogy_scoring_matches_compute_accuracy_casing_and_threshold() -> None:
    lookup = Lookup(
        {
            b"athens": (1, 0),
            b"greece": (1, 1),
            b"baghdad": (2, 0),
            b"iraq": (2, 1),
            b"decoy": (2, 0.9),
        }
    )
    questions = parse_analogy_questions(
        [b": capital-common-countries", b"Athens Greece Baghdad Iraq"]
    )

    included = evaluate_analogies(lookup, questions, vocabulary_limit=5, batch_size=1)
    excluded = evaluate_analogies(lookup, questions, vocabulary_limit=3, batch_size=1)

    assert included.overall.score == 1.0
    assert (included.overall.valid_count, included.overall.total_count) == (1, 1)
    assert (excluded.overall.valid_count, excluded.overall.total_count) == (0, 1)


def test_similarity_sentence_completion_and_oov_policies() -> None:
    lookup = Lookup(
        {
            b"hot": (1, 0),
            b"warm": (0.9, 0.1),
            b"cold": (-1, 0),
            b"sun": (1, 0),
            b"ice": (-1, 0),
        }
    )
    similarity = evaluate_word_similarity(
        lookup,
        [
            SimilarityPair(b"hot", b"warm", 3.0),
            SimilarityPair(b"hot", b"cold", 1.0),
            SimilarityPair(b"hot", b"missing", 2.0),
        ],
    )
    assert similarity.score == pytest.approx(1.0)
    assert similarity.coverage == 2 / 3

    completion = evaluate_sentence_completion(
        lookup,
        lookup.embeddings,
        [
            SentenceCompletionQuestion((b"sun", b"warm"), (b"hot", b"cold"), b"hot"),
            SentenceCompletionQuestion((b"missing",), (b"hot", b"cold"), b"hot"),
        ],
    )
    assert completion.score == 1.0
    assert (
        completion.total_count,
        completion.valid_count,
        completion.excluded_count,
    ) == (2, 1, 1)


def test_best_selection_and_report_are_deterministic() -> None:
    epoch_results = [
        (3, EvaluationResult("analogy_accuracy", 0.7, 10, 8, 2)),
        (2, EvaluationResult("analogy_accuracy", 0.7, 10, 9, 1)),
        (1, EvaluationResult("analogy_accuracy", 0.7, 10, 9, 1)),
    ]
    assert select_best_epoch(epoch_results)[0] == 1

    target = PaperTargetMetric("analogy_accuracy", 0.75)
    first = summarize_evaluations(
        [item[1] for item in epoch_results], condition_id="fixture", target=target
    )
    second = summarize_evaluations(
        [item[1] for item in epoch_results], condition_id="fixture", target=target
    )
    assert summary_record(first) == summary_record(second)
    assert summary_record(first)["paper_value"] == 0.75
    assert summary_record(first)["mean_coverage"] == (0.8 + 0.9 + 0.9) / 3
