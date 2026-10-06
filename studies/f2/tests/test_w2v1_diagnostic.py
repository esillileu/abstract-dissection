"""Tests for W2V1 Table 3 diagnostic evaluation results and aggregation invariants."""

from __future__ import annotations

import csv

from f2.suites.w2v1.diagnostic import _build_summary_markdown
from repro_core.context import RuntimePaths


def test_diagnostic_category_accuracy_csv_integrity() -> None:
    paths = RuntimePaths.from_environment()
    csv_path = paths.analysis_output("f2", "w2v1") / "diagnostic/category_accuracy.csv"
    if not csv_path.is_file():
        return

    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 17  # 14 categories + 3 aggregates

    # Check Total row
    total_row = next(r for r in rows if r["category"] == "Total")
    assert total_row["total_questions"] == "19544"
    assert total_row["included_questions"] == "18568"
    assert total_row["oov_questions"] == "976"
    assert float(total_row["cbow_accuracy_percent"]) == 34.89
    assert float(total_row["skipgram_accuracy_percent"]) == 55.95

    # Check Semantic Total row
    sem_row = next(r for r in rows if r["category"] == "Semantic Total")
    assert sem_row["total_questions"] == "8869"
    assert sem_row["included_questions"] == "8091"
    assert sem_row["oov_questions"] == "778"
    assert float(sem_row["cbow_accuracy_percent"]) == 13.67
    assert float(sem_row["skipgram_accuracy_percent"]) == 58.98

    # Check Syntactic Total row
    syn_row = next(r for r in rows if r["category"] == "Syntactic Total")
    assert syn_row["total_questions"] == "10675"
    assert syn_row["included_questions"] == "10477"
    assert syn_row["oov_questions"] == "198"
    assert float(syn_row["cbow_accuracy_percent"]) == 51.27
    assert float(syn_row["skipgram_accuracy_percent"]) == 53.62


def test_diagnostic_frequency_quantile_accuracy_csv_integrity() -> None:
    paths = RuntimePaths.from_environment()
    csv_path = (
        paths.analysis_output("f2", "w2v1")
        / "diagnostic/frequency_quantile_accuracy.csv"
    )
    if not csv_path.is_file():
        return

    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 8  # 4 semantic + 4 syntactic

    sem_rows = [r for r in rows if r["section"] == "semantic"]
    syn_rows = [r for r in rows if r["section"] == "syntactic"]

    assert sum(int(r["question_count"]) for r in sem_rows) == 8091
    assert sum(int(r["question_count"]) for r in syn_rows) == 10477

    # In syntactic, CBOW and SG are within 5% across all quantiles
    for r in syn_rows:
        diff = abs(float(r["accuracy_diff_percent"]))
        assert diff < 5.0


def test_diagnostic_top30k_csv_integrity() -> None:
    paths = RuntimePaths.from_environment()
    csv_path = paths.analysis_output("f2", "w2v1") / "diagnostic/top30k_diagnostic.csv"
    if not csv_path.is_file():
        return

    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 4

    cbow_full = next(
        r for r in rows if r["model"] == "CBOW" and r["eval_subset"] == "Full Table 3"
    )
    cbow_30k = next(
        r for r in rows if r["model"] == "CBOW" and r["eval_subset"] == "Top-30K-only"
    )

    assert float(cbow_full["total_accuracy_percent"]) == 34.89
    assert float(cbow_30k["total_accuracy_percent"]) == 41.51
    assert cbow_30k["included_questions"] == "14085"


def test_build_summary_markdown_generates_valid_text() -> None:
    text = _build_summary_markdown(
        category_rows=[
            {
                "category": "family",
                "section": "semantic",
                "total_questions": 506,
                "included_questions": 506,
                "oov_questions": 0,
                "min_frequency_median": 1118.0,
                "cbow_accuracy_percent": 58.17,
                "skipgram_accuracy_percent": 62.06,
                "accuracy_diff_percent": -3.89,
            }
        ],
        quantile_rows=[
            {
                "section": "semantic",
                "quantile": "Q1",
                "frequency_range": "[32, 159]",
                "question_count": 2067,
                "cbow_accuracy_percent": 7.40,
                "skipgram_accuracy_percent": 57.75,
                "accuracy_diff_percent": -50.35,
            }
        ],
        top30k_rows=[
            {
                "model": "CBOW",
                "eval_subset": "Full Table 3",
                "semantic_accuracy_percent": 13.67,
                "syntactic_accuracy_percent": 51.27,
                "total_accuracy_percent": 34.89,
                "included_questions": 18568,
                "total_questions": 19544,
                "coverage_ratio_percent": 95.01,
            }
        ],
    )
    assert "# W2V1 Table 3 CBOW vs Skip-gram Diagnostic Report" in text
    assert "CBOW Acc (%)" in text
