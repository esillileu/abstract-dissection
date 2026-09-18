"""Direct joint end-to-end evaluation and production pipeline recommendations."""

from __future__ import annotations

from typing import Any

from .models import ProductionPipelineRecommendation
from .rules import ALL_BINARY_EXT, DISQUALIFIED_PATH


def evaluate_production_recommendations(
    all_records: list[dict[str, Any]],
    audited_items: list[dict[str, Any]],
) -> list[ProductionPipelineRecommendation]:
    """Direct joint end-to-end evaluation on gold sample (without marginal multiplication)."""
    tot_gold_w = sum(
        r["total_weight"] * r["word_count_gold"]
        for r in audited_items
        if r["gold_class"] == 1
    )
    tot_gold_doc = sum(r["total_weight"] for r in audited_items if r["gold_class"] == 1)

    tot_10k_reqs = len(all_records)
    tot_10k_bytes = sum(float(r.get("downloaded_bytes", 0)) for r in all_records)
    all_ext_words = sum(
        r["total_weight"] * float(r["word_count"])
        for r in audited_items
        if int(r.get("extraction_success", 0)) == 1
        and int(r.get("is_english", 0)) == 1
        and int(r.get("is_valid", 0)) == 1
    )

    configs = [
        (
            "Baseline (Unfiltered Fetch + Default Post)",
            "None",
            lambda r: True,
            1.00,
        ),
        (
            "Leading Production Pipeline (Rule 1 Only + Post tau=1.25)",
            "Rule 1 Only (Binary & Media Exclusions)",
            lambda r: not ALL_BINARY_EXT.search(str(r.get("url", ""))),
            1.25,
        ),
        (
            "Candidate A: Extended Prefilter (Rules 1+2+3 + Post tau=1.25)",
            "Rules 1+2+3 (Binary Exts + Asset Paths + Tiny Stubs)",
            lambda r: (
                not (
                    ALL_BINARY_EXT.search(str(r.get("url", "")))
                    or DISQUALIFIED_PATH.search(str(r.get("url", "")))
                    or (
                        0
                        < int(r.get("arc_length", r.get("downloaded_bytes", 0)))
                        < 1200
                    )
                )
            ),
            1.25,
        ),
        (
            "Candidate B: Conservative Threshold (Rule 1 Only + Post tau=1.00)",
            "Rule 1 Only (Binary & Media Exclusions)",
            lambda r: not ALL_BINARY_EXT.search(str(r.get("url", ""))),
            1.00,
        ),
    ]

    recs: list[ProductionPipelineRecommendation] = []
    for name, desc, pre_fn, post_tau in configs:
        passed_10k = [r for r in all_records if pre_fn(r)]
        reqs_saved = 1.0 - (len(passed_10k) / tot_10k_reqs)
        bytes_passed = sum(float(r.get("downloaded_bytes", 0)) for r in passed_10k)
        bytes_saved = 1.0 - (bytes_passed / tot_10k_bytes) if tot_10k_bytes > 0 else 0.0

        tp_joint_w = 0.0
        tp_joint_doc = 0.0
        fp_joint_w = 0.0
        fp_joint_doc = 0.0
        retained_words = 0.0

        for r in audited_items:
            pre_ok = pre_fn(r)
            post_ok = (
                int(r.get("is_english", 0)) == 1
                and int(r.get("is_valid", 0)) == 1
                and float(r.get("news_score", 0.0)) >= post_tau
            )
            joint_ok = pre_ok and post_ok

            if joint_ok:
                retained_words += r["total_weight"] * float(r.get("word_count", 0))
                if r["gold_class"] == 1:
                    tp_joint_w += r["total_weight"] * r["word_count_gold"]
                    tp_joint_doc += r["total_weight"]
                else:
                    fp_joint_w += r["total_weight"] * float(r.get("word_count", 0))
                    fp_joint_doc += r["total_weight"]

        joint_w_rec = tp_joint_w / tot_gold_w if tot_gold_w > 0 else 0.0
        joint_d_rec = tp_joint_doc / tot_gold_doc if tot_gold_doc > 0 else 0.0
        joint_w_ppv = (
            tp_joint_w / (tp_joint_w + fp_joint_w)
            if (tp_joint_w + fp_joint_w) > 0
            else 0.0
        )
        joint_d_ppv = (
            tp_joint_doc / (tp_joint_doc + fp_joint_doc)
            if (tp_joint_doc + fp_joint_doc) > 0
            else 0.0
        )

        disk_saved = (
            1.0 - (retained_words / all_ext_words) if all_ext_words > 0 else 0.0
        )

        base_true_words = 521_357_336_694.0
        net_words_15 = base_true_words * joint_w_rec * 0.85
        net_words_50 = base_true_words * joint_w_rec * 0.50
        margin_50 = net_words_50 / 33_000_000_000.0

        recs.append(
            ProductionPipelineRecommendation(
                config_name=name,
                prefetch_rule_desc=desc,
                postfetch_threshold=post_tau,
                joint_word_recall=joint_w_rec,
                joint_doc_recall=joint_d_rec,
                joint_word_ppv=joint_w_ppv,
                joint_doc_ppv=joint_d_ppv,
                avoided_arc_requests_pct=reqs_saved,
                avoided_network_bytes_pct=bytes_saved,
                avoided_disk_storage_pct=disk_saved,
                projected_net_words_15pct_dedup=net_words_15,
                projected_net_words_50pct_dedup=net_words_50,
                margin_vs_33b_50pct_dedup=margin_50,
            )
        )
    return recs


__all__ = ["evaluate_production_recommendations"]
