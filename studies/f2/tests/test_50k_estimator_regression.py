"""Regression tests verifying 50k confirmatory estimator fidelity, 8-stratum redistribution, and reject-subsampling variance inflation."""

from __future__ import annotations

import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from f2.analysis import FeasibilityAnalyzer
from f2.discovery import SequentialAuditSampler


def test_8_stratum_sparse_cell_redistribution():
    """Verify that 8-stratum audit wave selection handles sparse cells and deterministically redistributes quota."""
    sampler = SequentialAuditSampler(seed=20260227)

    mock_schedule = []
    # S1: 200 records
    for i in range(200):
        mock_schedule.append(
            {
                "candidate_id": f"s1_{i}",
                "record_id": f"s1_{i}",
                "url": f"http://s1/{i}",
                "crawl_id": "CC-MAIN-2009-2010",
                "prefilter_status": "pass",
                "is_news_predicted": 1,
                "design_stratum": "S1",
                "stratum_total": 200,
                "priority_order": i,
            }
        )
    # S2: 100 records
    for i in range(100):
        mock_schedule.append(
            {
                "candidate_id": f"s2_{i}",
                "record_id": f"s2_{i}",
                "url": f"http://s2/{i}",
                "crawl_id": "CC-MAIN-2009-2010",
                "prefilter_status": "pass",
                "is_news_predicted": 0,
                "design_stratum": "S2",
                "stratum_total": 100,
                "priority_order": i,
            }
        )
    # S3: only 3 records (sparse!)
    for i in range(3):
        mock_schedule.append(
            {
                "candidate_id": f"s3_{i}",
                "record_id": f"s3_{i}",
                "url": f"http://s3/{i}",
                "crawl_id": "CC-MAIN-2009-2010",
                "prefilter_status": "reject",
                "is_news_predicted": 1,
                "design_stratum": "S3",
                "stratum_total": 3,
                "priority_order": i,
            }
        )
    # S4: 20 records (companion reject)
    for i in range(20):
        mock_schedule.append(
            {
                "candidate_id": f"s4_{i}",
                "record_id": f"s4_{i}",
                "url": f"http://s4/{i}",
                "crawl_id": "CC-MAIN-2009-2010",
                "prefilter_status": "reject",
                "is_news_predicted": 0,
                "design_stratum": "S4",
                "stratum_total": 20,
                "priority_order": i,
            }
        )
    # S5: 200 records
    for i in range(200):
        mock_schedule.append(
            {
                "candidate_id": f"s5_{i}",
                "record_id": f"s5_{i}",
                "url": f"http://s5/{i}",
                "crawl_id": "CC-MAIN-2012",
                "prefilter_status": "pass",
                "is_news_predicted": 1,
                "design_stratum": "S5",
                "stratum_total": 200,
                "priority_order": i,
            }
        )
    # S6: 100 records
    for i in range(100):
        mock_schedule.append(
            {
                "candidate_id": f"s6_{i}",
                "record_id": f"s6_{i}",
                "url": f"http://s6/{i}",
                "crawl_id": "CC-MAIN-2012",
                "prefilter_status": "pass",
                "is_news_predicted": 0,
                "design_stratum": "S6",
                "stratum_total": 100,
                "priority_order": i,
            }
        )
    # S7: only 4 records (sparse!)
    for i in range(4):
        mock_schedule.append(
            {
                "candidate_id": f"s7_{i}",
                "record_id": f"s7_{i}",
                "url": f"http://s7/{i}",
                "crawl_id": "CC-MAIN-2012",
                "prefilter_status": "reject",
                "is_news_predicted": 1,
                "design_stratum": "S7",
                "stratum_total": 4,
                "priority_order": i,
            }
        )
    # S8: 20 records (companion reject)
    for i in range(20):
        mock_schedule.append(
            {
                "candidate_id": f"s8_{i}",
                "record_id": f"s8_{i}",
                "url": f"http://s8/{i}",
                "crawl_id": "CC-MAIN-2012",
                "prefilter_status": "reject",
                "is_news_predicted": 0,
                "design_stratum": "S8",
                "stratum_total": 20,
                "priority_order": i,
            }
        )

    selected = sampler.select_8_stratum_audit_wave(mock_schedule, total_budget=400)
    assert len(selected) == 400

    by_strat: dict[str, int] = {}
    for item in selected:
        by_strat[item["design_stratum"]] = by_strat.get(item["design_stratum"], 0) + 1

    assert by_strat["S3"] == 3
    assert by_strat["S7"] == 4
    assert by_strat["S4"] == 17
    assert by_strat["S8"] == 16

    for item in selected:
        sid = item["design_stratum"]
        expected_pi = by_strat[sid] / item["stratum_total"]
        assert abs(item["audit_inclusion_prob_cond"] - expected_pi) < 1e-6
        assert abs(item["audit_weight_cond"] - 1.0 / expected_pi) < 1e-6


def test_reject_subsampling_variance_inflation(tmp_path: Path):
    """Verify that 5% reject-exploration subsampling appropriately increases bootstrap variance compared to 100% fetch."""
    base_pi = 1e-6
    rng = random.Random(42)

    # 1. Full Dataset A (100% fetch: 50 pass + 50 reject all fetched)
    rows_full = []
    for b in range(10):
        for r in range(10):
            rec_id = f"full_{b}_{r}"
            is_rej = r % 2 == 1
            pref = "reject" if is_rej else "pass"
            f_prob = 1.0
            w = 1.0 / (base_pi * f_prob)
            proxy_w = 500 if not is_rej else 0
            rows_full.append(
                {
                    "record_id": rec_id,
                    "crawl_id": "CC-MAIN-2012",
                    "url": f"http://test.com/{rec_id}.pdf"
                    if is_rej
                    else f"http://test.com/{rec_id}.html",
                    "inclusion_probability": base_pi * f_prob,
                    "design_weight": w,
                    "block_index": b,
                    "record_index_in_block": r,
                    "block_total_records": 10,
                    "prefilter_status": pref,
                    "prefilter_rule": "rule1" if is_rej else "none",
                    "fetch_probability": f_prob,
                    "is_selected_for_fetch": 1,
                    "fetch_status": "success",
                    "http_status": 200,
                    "downloaded_bytes": 5000,
                    "extraction_success": 1,
                    "news_score": 1.5 if not is_rej else 0.5,
                    "is_news_predicted": 1 if not is_rej else 0,
                    "is_english": 1,
                    "is_valid": 1,
                    "rejection_reason": None,
                    "word_count": 500,
                    "proxy_words": proxy_w,
                    "clean_text_sha256": "hash",
                    "shard_path": "shard_0.txt",
                    "diagnostics": "{}",
                    "is_audited": 1,
                    "gold_class": 1,
                    "word_count_gold": 500,
                    "word_residual": 500 if is_rej else 0,
                    "audit_inclusion_probability": 1.0,
                    "audit_design_weight": 1.0,
                    "design_stratum": "S5" if not is_rej else "S8",
                }
            )

    # 2. Subsampled Dataset B (5% reject fetch: 50 pass + 3 sampled rejects fetched with 20x weight)
    rows_sub = []
    # 50 pass records
    for b in range(10):
        for r in range(0, 10, 2):
            rec_id = f"sub_pass_{b}_{r}"
            rows_sub.append(
                {
                    "record_id": rec_id,
                    "crawl_id": "CC-MAIN-2012",
                    "url": f"http://test.com/{rec_id}.html",
                    "inclusion_probability": base_pi,
                    "design_weight": 1.0 / base_pi,
                    "block_index": b,
                    "record_index_in_block": r,
                    "block_total_records": 10,
                    "prefilter_status": "pass",
                    "prefilter_rule": "none",
                    "fetch_probability": 1.0,
                    "is_selected_for_fetch": 1,
                    "fetch_status": "success",
                    "http_status": 200,
                    "downloaded_bytes": 5000,
                    "extraction_success": 1,
                    "news_score": 1.5,
                    "is_news_predicted": 1,
                    "is_english": 1,
                    "is_valid": 1,
                    "rejection_reason": None,
                    "word_count": 500,
                    "proxy_words": 500,
                    "clean_text_sha256": "hash",
                    "shard_path": "shard_0.txt",
                    "diagnostics": "{}",
                    "is_audited": 1,
                    "gold_class": 1,
                    "word_count_gold": 500,
                    "word_residual": 0,
                    "audit_inclusion_probability": 1.0,
                    "audit_design_weight": 1.0,
                    "design_stratum": "S5",
                }
            )

    # Sample exactly 2-3 reject records (e.g. from blocks 1 and 5) with f_prob = 0.05 (20x weight)
    for b in [1, 5]:
        rec_id = f"sub_rej_{b}_1"
        f_prob = 0.05
        rows_sub.append(
            {
                "record_id": rec_id,
                "crawl_id": "CC-MAIN-2012",
                "url": f"http://test.com/{rec_id}.pdf",
                "inclusion_probability": base_pi * f_prob,
                "design_weight": 1.0 / (base_pi * f_prob),
                "block_index": b,
                "record_index_in_block": 1,
                "block_total_records": 10,
                "prefilter_status": "reject",
                "prefilter_rule": "rule1",
                "fetch_probability": f_prob,
                "is_selected_for_fetch": 1,
                "fetch_status": "success",
                "http_status": 200,
                "downloaded_bytes": 5000,
                "extraction_success": 1,
                "news_score": 0.5,
                "is_news_predicted": 0,
                "is_english": 1,
                "is_valid": 1,
                "rejection_reason": None,
                "word_count": 500,
                "proxy_words": 0,
                "clean_text_sha256": "hash",
                "shard_path": "shard_0.txt",
                "diagnostics": "{}",
                "is_audited": 1,
                "gold_class": 1,
                "word_count_gold": 500,
                "word_residual": 500,
                "audit_inclusion_probability": 1.0,
                "audit_design_weight": 1.0,
                "design_stratum": "S8",
            }
        )

    path_full = tmp_path / "prov_full.parquet"
    path_sub = tmp_path / "prov_sub.parquet"
    pq.write_table(pa.Table.from_pylist(rows_full), path_full.as_posix())
    pq.write_table(pa.Table.from_pylist(rows_sub), path_sub.as_posix())

    audits_full = [
        {
            "record_id": r["record_id"],
            "word_count_gold": r["word_count_gold"],
            "gold_class": r["gold_class"],
        }
        for r in rows_full
    ]
    audits_sub = [
        {
            "record_id": r["record_id"],
            "word_count_gold": r["word_count_gold"],
            "gold_class": r["gold_class"],
        }
        for r in rows_sub
    ]

    analyzer_full = FeasibilityAnalyzer(path_full)
    analyzer_sub = FeasibilityAnalyzer(path_sub)

    res_full = analyzer_full.compute_two_phase_yield(
        audit_records=audits_full, bootstrap_reps=500, seed=42
    )
    res_sub = analyzer_sub.compute_two_phase_yield(
        audit_records=audits_sub, bootstrap_reps=500, seed=42
    )

    # Variance under 5% reject exploration subsampling must be strictly greater than under 100% fetch
    assert res_sub.aggregated_std_error > res_full.aggregated_std_error
