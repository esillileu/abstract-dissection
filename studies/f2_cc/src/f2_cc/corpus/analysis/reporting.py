"""Publication-grade markdown feasibility report generator."""

from __future__ import annotations

from .models import FeasibilityReportData


def generate_report_markdown(data: FeasibilityReportData) -> str:
    """Render a publication-grade markdown feasibility report."""
    audit_tag = (
        f"**Audit-Corrected ({data.audit_sample_size} Gold Audits across 8 Design Strata)**"
        if data.has_audit
        else "**Uncorrected Proxy (No Phase-2 Audit)**"
    )
    verdict_str = "FEASIBLE" if data.has_audit else "PROVISIONALLY FEASIBLE"
    lines = [
        "# Common Crawl (2009-2012) 50,000 Confirmatory Feasibility Report",
        "",
        f"**Estimation Mode:** {audit_tag}",
        "**Variance Method:** Two-Stage Cluster & Reject Subsampling Bootstrap (Resampling CDX Blocks, Records, and 8-Stratum Residuals)",
        "**Baseline 10k Calibration SHA:** `f3dee9676517d9a7506b162aff83a111f45209dc`",
        "",
        "## 1. Executive Summary & Scale Feasibility Verdicts",
        "",
        "| Target Corpus Scale | Feasibility Verdict | Minimum Projected Words (95% CI Lower) | Safety Margin vs Target |",
        "| :--- | :--- | :--- | :--- |",
        f"| **1 Billion Words (1B)** | **{verdict_str if data.feasibility_1b else 'INSUFFICIENT'}** | {data.aggregated_ci_lower_95:,.0f} words | {data.aggregated_ci_lower_95 / 1_000_000_000:.1f}x |",
        f"| **6 Billion Words (6B)** | **{verdict_str if data.feasibility_6b else 'INSUFFICIENT'}** | {data.aggregated_ci_lower_95:,.0f} words | {data.aggregated_ci_lower_95 / 6_000_000_000:.1f}x |",
        f"| **33 Billion Words (33B)** | **{verdict_str if data.feasibility_33b else 'INSUFFICIENT'}** | {data.aggregated_ci_lower_95:,.0f} words | {data.aggregated_ci_lower_95 / 33_000_000_000:.1f}x |",
        "",
        "---",
        "",
        "## 2. Feasibility Replication & 10k Baseline Comparison",
        "",
        "| Metric Layer | 10k Design/Tuning Baseline | 50k Confirmatory Run | Consistency Diagnostic Status |",
        "| :--- | :--- | :--- | :--- |",
        f"| **True Total News Words ($\\hat{{W}}_{{\\text{{true}}}}$)** | 521.36 Billion words | **{data.aggregated_true_words / 1e9:.2f} Billion words** | {'Consistent (Inside 10k CI)' if data.baseline_10k_comparison and data.baseline_10k_comparison['is_inside_10k_ci'] else 'Diagnostic Observation'} |",
        f"| **95% Bootstrap Confidence Interval** | [368.13B, 674.58B] words | **[{data.aggregated_ci_lower_95 / 1e9:.2f}B, {data.aggregated_ci_upper_95 / 1e9:.2f}B] words** | Shrinkage Monitored |",
        f"| **Relative Standard Error (RSE)** | 15.1% | **{data.aggregated_std_error / max(1.0, data.aggregated_true_words) * 100:.1f}%** | Diagnostic Monitored |",
        f"| **50% Dedup Conservative Margin vs 33B** | 5.56x | **{(data.aggregated_ci_lower_95 * 0.50) / 33_000_000_000:.2f}x** | {'FEASIBLE (>= 3.0x)' if (data.aggregated_ci_lower_95 * 0.50) / 33_000_000_000 >= 3.0 else 'Monitored'} |",
        "",
        "---",
        "",
        "## 3. Statistical Yield & Residual Estimates Across Crawl Strata",
        "",
        "| Crawl Stratum | Sample Size | Retained Docs | Uncorrected Proxy Words | Residual Error ($\\hat{E}_c$) | True Total Words (95% CI) | Weighted PPV | Weighted TPR |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for y in data.strata_yields:
        res_str = (
            f"{y.residual_error_words:+,.0f}" if data.has_audit else "0 (No Audit)"
        )
        ppv_s = f"{y.weighted_ppv * 100:.1f}%" if y.weighted_ppv is not None else "N/A"
        tpr_s = f"{y.weighted_tpr * 100:.1f}%" if y.weighted_tpr is not None else "N/A"
        lines.append(
            f"| **{y.crawl_id}** | {y.sample_size:,} | {y.retained_news_docs:,} | {y.proxy_total_words:,.0f} | {res_str} | **{y.true_total_words:,.0f}** [{y.ci_lower_95:,.0f}, {y.ci_upper_95:,.0f}] | {ppv_s} | {tpr_s} |"
        )

    lines.extend(
        [
            f"| **Aggregated Total** | — | — | — | — | **{data.aggregated_true_words:,.0f}** [{data.aggregated_ci_lower_95:,.0f}, {data.aggregated_ci_upper_95:,.0f}] | — | — |",
            "",
            "---",
            "",
        ]
    )

    # 4. Sequential Funnel
    if data.sequential_funnel:
        lines.extend(
            [
                "## 4. End-to-End Pipeline Funnel (Strictly Monotonic Survival)",
                "",
                "| Crawl Stratum | Step 0: Sampled | Step 1: Fetch OK | Step 2: Extraction OK | Step 3: News Pred | Step 4: English News | Step 5: Retained Valid News | Avg Words | Median Words |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for f in data.sequential_funnel:
            lines.append(
                f"| **{f['crawl_id']}** | {int(f['step0_sampled']):,} | {int(f['step1_fetch_ok']):,} | {int(f['step2_extraction_ok']):,} | {int(f['step3_news_pred']):,} | {int(f['step4_english_news']):,} | **{int(f['step5_retained_valid_news']):,}** | {float(f['avg_words_per_doc'] or 0):.1f} | {float(f['median_words_per_doc'] or 0):.1f} |"
            )
        lines.extend(["", "---", ""])

    # 5. Marginal Filters
    if data.marginal_filters:
        lines.extend(
            [
                "## 5. Independent Marginal Filter Pass Rates (Across All Extracted Documents)",
                "",
                "| Crawl Stratum | Extracted Docs | Marginal News Filter | Marginal English Filter | Marginal Format/Length Filter |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for m in data.marginal_filters:
            tot = int(m["total_extracted"])
            lines.append(
                f"| **{m['crawl_id']}** | {tot:,} | {int(m['marginal_news_pred']):,} ({int(m['marginal_news_pred']) / max(1, tot) * 100:.1f}%) | {int(m['marginal_english_pass']):,} ({int(m['marginal_english_pass']) / max(1, tot) * 100:.1f}%) | {int(m['marginal_valid_pass']):,} ({int(m['marginal_valid_pass']) / max(1, tot) * 100:.1f}%) |"
            )
        lines.extend(["", "---", ""])

    # 6. Deduplication scenarios
    lines.extend(
        [
            "## 6. Deduplication Sensitivity Scenarios (Net Word Yield & 95% Confidence Bounds)",
            "",
            "| Scenario Description | Assumed Duplicate Rate | Projected Net Words (Point Est) | Net 95% Bootstrap CI | Point Safety Margin vs 33B | Conservative Safety Margin (95% Lower) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for s in data.dedup_scenarios:
        lines.append(
            f"| **{s.name}** | {s.dedup_rate * 100:.0f}% | {s.net_point_words:,.0f} words | [{s.net_ci_lower_95:,.0f}, {s.net_ci_upper_95:,.0f}] | {s.point_margin_vs_33b:.1f}x | **{s.lower_margin_vs_33b:.2f}x** |"
        )

    lines.extend(["", "---", ""])

    # 7. Methodological & Diagnostic Metrics
    ppv_str = (
        f"{data.precision_ppv * 100:.1f}%" if data.precision_ppv is not None else "N/A"
    )
    tpr_str = f"{data.recall_tpr * 100:.1f}%" if data.recall_tpr is not None else "N/A"
    lines.extend(
        [
            "## 7. Methodological & Diagnostic Metrics",
            "",
            f"* **Phase-2 Probability Audit Sample**: {data.audit_sample_size} audited documents across 8 design strata",
            f"* **Overall Classifier Precision (PPV)**: {ppv_str}",
            f"* **Overall Classifier Recall (TPR)**: {tpr_str}",
            r"* **Estimation Method**: Two-Phase Stratified Difference Estimator ($\hat{W}_{\text{true}} = \hat{W}_{\text{proxy}} + \hat{E}$)",
            f"* **Good-Turing Domain Coverage**: {data.good_turing_coverage * 100:.2f}%",
            f"* **Chao1 Estimated Publisher Richness**: {data.chao1_richness:,.0f} domains",
            "",
        ]
    )

    return "\n".join(lines)
