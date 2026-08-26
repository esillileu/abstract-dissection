"""Tests for FeasibilityAnalyzer, Two-Stage Horvitz-Thompson estimation, two-phase residual difference correction, and deduplication scenarios."""

from __future__ import annotations

import json
from pathlib import Path

from f2.analysis import FeasibilityAnalyzer


def test_feasibility_analyzer_estimation(tmp_path: Path):
    prov_path = tmp_path / "provenance.jsonl"
    records = []

    # Mock 100 records for CC-MAIN-2012 (w_i = 10,000,000)
    for i in range(100):
        is_news = 1 if i < 30 else 0
        words = 500 if is_news else 0
        records.append(
            {
                "record_id": f"rec_{i}",
                "crawl_id": "CC-MAIN-2012",
                "url": f"http://test{i}.com",
                "fetch_status": "success",
                "downloaded_bytes": 10000,
                "http_status": 200,
                "extraction_success": 1,
                "news_score": 2.0 if is_news else 0.0,
                "is_news_predicted": is_news,
                "is_english": 1,
                "is_valid": 1,
                "rejection_reason": None,
                "word_count": words,
                "inclusion_probability": 0.0001,
                "design_weight": 10000.0,
                "proxy_words": words,
                "diagnostics": {},
            }
        )

    with prov_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    analyzer = FeasibilityAnalyzer(prov_path)
    funnel = analyzer.compute_funnel_summary()
    assert funnel["total_sampled"] == 100
    assert funnel["valid_news"] == 30
    assert funnel["avg_words_per_doc"] == 500.0

    # Test audit correction with gold residuals (e.g. 5 audited records)
    audit_records = [
        {
            "record_id": "rec_0",
            "predicted_class": 1,
            "gold_class": 1,
            "gold_words": 500,
        },
        {
            "record_id": "rec_1",
            "predicted_class": 1,
            "gold_class": 1,
            "gold_words": 550,
        },  # +50 residual
        {"record_id": "rec_30", "predicted_class": 0, "gold_class": 0, "gold_words": 0},
        {
            "record_id": "rec_31",
            "predicted_class": 0,
            "gold_class": 1,
            "gold_words": 400,
        },  # +400 false negative residual
    ]

    report_data = analyzer.compute_two_phase_yield(
        audit_records=audit_records, bootstrap_reps=50, seed=42
    )
    assert len(report_data.strata_yields) == 1
    yield_2012 = report_data.strata_yields[0]

    # 30 news * 500 words * 10,000 weight = 150,000,000 proxy words
    assert yield_2012.proxy_total_words == 150_000_000.0
    assert (
        yield_2012.true_total_words > 150_000_000.0
    )  # Increased due to positive residual in audit
    assert report_data.aggregated_true_words == yield_2012.true_total_words

    # Test deduplication scenarios
    assert (
        report_data.scenarios["exact_15pct"] == report_data.aggregated_true_words * 0.85
    )
    assert (
        report_data.scenarios["moderate_syndication_30pct"]
        == report_data.aggregated_true_words * 0.70
    )

    # Markdown report generation check
    md = analyzer.generate_report_markdown(report_data)
    assert "# Common Crawl (2009-2012) Corpus Feasibility Study Report" in md
    assert "Deduplication Sensitivity Scenarios" in md
