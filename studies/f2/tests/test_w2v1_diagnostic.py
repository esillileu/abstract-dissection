"""Question-level diagnostic scoring must retain the aggregate evaluator protocol."""

import numpy as np

from f2.suites.w2v.evaluation import _analogy_predictions, _normalized_embeddings
from f2.suites.w2v1.diagnostic import _build_summary_markdown


def test_predictions_exclude_inputs_and_require_positive_score():
    vectors = _normalized_embeddings(
        np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0]], dtype=np.float32),
        dtype=np.float32,
    )
    prediction, positive = next(_analogy_predictions(vectors, [(0, 1, 2, 3)], 1))
    assert prediction == 3 and positive
    zero = np.zeros((4, 2), dtype=np.float32)
    prediction, positive = next(_analogy_predictions(zero, [(0, 1, 2, 3)], 1))
    assert prediction == 3 and not positive


def test_report_uses_computed_results_without_fixed_claims():
    rows = [
        {
            "category": "fixture",
            "section": "semantic",
            "included_questions": 2,
            "cbow_accuracy_percent": 12.345,
            "skipgram_accuracy_percent": 20.0,
            "accuracy_diff_percent": -7.655,
        }
    ]
    quantiles = [
        dict(rows[0], section=section) for section in ("semantic", "syntactic")
    ]
    subsets = [
        dict(
            model=model,
            eval_subset=subset,
            semantic_accuracy_percent=12.345,
            syntactic_accuracy_percent=20.0,
            total_accuracy_percent=16.1725,
            syntactic_included_questions=1,
            included_questions=2,
        )
        for model in ("CBOW", "Skip-gram")
        for subset in ("Full Table 3", "Top-30K-only")
    ]
    report = _build_summary_markdown(rows, quantiles, subsets)
    assert "12.35" in report
    assert "13.67" not in report
    assert "capital-world" not in report
