"""Tests for FeasibilityAnalyzer, Two-Stage Horvitz-Thompson estimation, two-phase residual difference correction, and deduplication scenarios."""

from __future__ import annotations

import json
from pathlib import Path

from f2.analysis import FeasibilityAnalyzer


def test_feasibility_analyzer_estimation_with_and_without_audit(tmp_path: Path):
    prov_path = tmp_path / "provenance.jsonl"
    records = []

    # Mock 100 records for CC-MAIN-2012 (w_i = 10,000)
    for i in range(100):
        is_news = 1 if i < 30 else 0
        is_en = (
            1 if (i < 25 or (30 <= i < 60)) else 0
        )  # 25 of news are English, 30 of non-news are English
        is_val = 1 if (i < 20 or (30 <= i < 70)) else 0  # 20 of English news are valid
        words = 500 if (is_news and is_en and is_val) else 0
        records.append(
            {
                "record_id": f"rec_{i}",
                "crawl_id": "CC-MAIN-2012",
                "url": f"http://test{i}.com",
                "fetch_status": "success" if i < 95 else "failed",
                "downloaded_bytes": 10000 if i < 95 else 0,
                "http_status": 200 if i < 95 else 500,
                "extraction_success": 1 if i < 90 else 0,
                "news_score": 2.0 if is_news else 0.0,
                "is_news_predicted": is_news,
                "is_english": is_en,
                "is_valid": is_val,
                "rejection_reason": None if words > 0 else "filtered",
                "word_count": 500 if is_val else 50,
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

    # 1. Test strictly monotonic sequential funnel
    seq_funnel = analyzer.compute_sequential_funnel()
    assert len(seq_funnel) == 1
    f = seq_funnel[0]
    assert f["step0_sampled"] == 100
    assert f["step1_fetch_ok"] == 95
    assert f["step2_extraction_ok"] == 90
    assert f["step3_news_pred"] <= f["step2_extraction_ok"]
    assert f["step4_english_news"] <= f["step3_news_pred"]
    assert f["step5_retained_valid_news"] <= f["step4_english_news"]
    assert f["step5_retained_valid_news"] == 20
    assert (
        f["step0_sampled"]
        >= f["step1_fetch_ok"]
        >= f["step2_extraction_ok"]
        >= f["step3_news_pred"]
        >= f["step4_english_news"]
        >= f["step5_retained_valid_news"]
    ), "Sequential funnel must be strictly monotonic non-increasing!"

    # 2. Test marginal filters (can be independent)
    marginal = analyzer.compute_marginal_filters()
    assert len(marginal) == 1
    m = marginal[0]
    assert m["total_extracted"] == 90

    # 3. Test estimation WITHOUT audit file (Uncorrected Proxy)
    data_no_audit = analyzer.compute_two_phase_yield(
        audit_records=None, bootstrap_reps=30, seed=42
    )
    assert not data_no_audit.has_audit
    assert data_no_audit.precision_ppv is None
    assert data_no_audit.recall_tpr is None
    assert data_no_audit.strata_yields[0].residual_error_words == 0.0
    # 20 docs * 500 words * 10,000 weight = 100,000,000 proxy words
    assert data_no_audit.strata_yields[0].proxy_total_words == 100_000_000.0
    assert data_no_audit.strata_yields[0].true_total_words == 100_000_000.0

    md_no_audit = analyzer.generate_report_markdown(data_no_audit)
    assert "Uncorrected Proxy" in md_no_audit
    assert "0 (No Audit)" in md_no_audit
    assert "End-to-End Pipeline Funnel (Strictly Monotonic Survival)" in md_no_audit
    assert "Independent Marginal Filter Pass Rates" in md_no_audit

    # 4. Test estimation WITH audit file (Residual Correction)
    audit_records = [
        {
            "record_id": "rec_0",
            "predicted_class": 1,
            "gold_class": 1,
            "word_count_gold": 500,
        },
        {
            "record_id": "rec_1",
            "predicted_class": 1,
            "gold_class": 1,
            "word_count_gold": 600,
        },  # +100 residual
        {
            "record_id": "rec_2",
            "predicted_class": 1,
            "gold_class": 0,
            "word_count_gold": 0,
        },  # -500 false positive residual
        {
            "record_id": "rec_30",
            "predicted_class": 0,
            "gold_class": 0,
            "word_count_gold": 0,
        },
        {
            "record_id": "rec_31",
            "predicted_class": 0,
            "gold_class": 1,
            "word_count_gold": 400,
        },  # +400 false negative residual
    ]
    # Pad to >= 10 for diagnostics
    for k in range(32, 40):
        audit_records.append(
            {
                "record_id": f"rec_{k}",
                "predicted_class": 0,
                "gold_class": 0,
                "word_count_gold": 0,
            }
        )

    data_with_audit = analyzer.compute_two_phase_yield(
        audit_records=audit_records, bootstrap_reps=50, seed=42
    )
    assert data_with_audit.has_audit
    assert data_with_audit.audit_sample_size == len(audit_records)
    assert data_with_audit.precision_ppv is not None
    assert data_with_audit.recall_tpr is not None
    assert data_with_audit.strata_yields[0].residual_error_words != 0.0
    assert data_with_audit.strata_yields[0].true_total_words == max(
        0.0,
        data_with_audit.strata_yields[0].proxy_total_words
        + data_with_audit.strata_yields[0].residual_error_words,
    )

    md_with_audit = analyzer.generate_report_markdown(data_with_audit)
    assert "Audit-Corrected" in md_with_audit
    assert "Classifier Precision (PPV)" in md_with_audit


def test_calibration_and_prefetch_analyzer(tmp_path: Path):
    from f2.calibration import CalibrationAndPreFetchAnalyzer

    prov_path = tmp_path / "provenance.jsonl"
    records = []
    audit_records = []

    for i in range(50):
        is_news = 1 if i < 20 else 0
        g_cls = 1 if (i < 15 or i == 25) else 0
        records.append(
            {
                "record_id": f"rec_{i}",
                "crawl_id": "CC-MAIN-2012",
                "url": f"http://news.example.com/2012/05/article-{i}.html"
                if is_news
                else f"http://store.example.com/product/{i}",
                "fetch_status": "success",
                "downloaded_bytes": 15000 if is_news else 2000,
                "arc_length": 8000 if is_news else 1000,
                "http_status": 200,
                "extraction_success": 1,
                "news_score": 2.5 if is_news else 0.0,
                "is_news_predicted": is_news,
                "is_english": 1,
                "is_valid": 1,
                "rejection_reason": None,
                "word_count": 600 if is_news else 50,
                "inclusion_probability": 0.0001,
                "design_weight": 10000.0,
                "proxy_words": 600 if is_news else 0,
                "diagnostics": {},
            }
        )
        audit_records.append(
            {
                "record_id": f"rec_{i}",
                "predicted_class": is_news,
                "gold_class": g_cls,
                "word_count_gold": 600 if g_cls else 0,
            }
        )

    with prov_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    calibrator = CalibrationAndPreFetchAnalyzer(prov_path, audit_records)
    post_sweep = calibrator.evaluate_postfetch_sweep()
    assert len(post_sweep) > 0

    pre_sweep = calibrator.evaluate_prefetch_sweep()
    assert len(pre_sweep) > 0

    recs = calibrator.evaluate_production_recommendations()
    assert len(recs) == 4

    md = calibrator.generate_report_markdown()
    assert "# Offline Classifier Calibration & Pre-Fetch Feasibility Study" in md
    assert "Recommended Balanced Pipeline" in md
