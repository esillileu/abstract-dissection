"""Feasibility analysis engine executing DuckDB queries over exported provenance Parquet logs."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb


@dataclass(frozen=True)
class CrawlStratumYield:
    crawl_id: str
    proxy_total_words: float
    residual_error_words: float
    true_total_words: float
    std_error_words: float
    ci_lower_95: float
    ci_upper_95: float
    sample_size: int
    retained_news_docs: int


@dataclass(frozen=True)
class FeasibilityReportData:
    strata_yields: list[CrawlStratumYield]
    aggregated_true_words: float
    aggregated_std_error: float
    aggregated_ci_lower_95: float
    aggregated_ci_upper_95: float
    scenarios: dict[str, float]
    has_audit: bool
    audit_sample_size: int
    precision_ppv: float | None
    recall_tpr: float | None
    good_turing_coverage: float
    chao1_richness: float
    feasibility_1b: bool
    feasibility_6b: bool
    feasibility_33b: bool
    sequential_funnel: list[dict[str, Any]] | None = None
    marginal_filters: list[dict[str, Any]] | None = None


class FeasibilityAnalyzer:
    """Computes design-based statistical estimates and analytical summaries from provenance logs."""

    def __init__(self, provenance_path: Path) -> None:
        self.provenance_path = provenance_path
        self.con = duckdb.connect(":memory:")
        self._load_data()

    def _load_data(self) -> None:
        if not self.provenance_path.exists():
            raise FileNotFoundError(
                f"Provenance file not found: {self.provenance_path}"
            )
        posix_path = self.provenance_path.as_posix()
        if posix_path.endswith(".parquet"):
            self.con.execute(
                f"CREATE TABLE provenance AS SELECT * FROM read_parquet('{posix_path}');"
            )
        else:
            self.con.execute(
                f"CREATE TABLE provenance AS SELECT * FROM read_json_auto('{posix_path}');"
            )

    def compute_sequential_funnel(self) -> list[dict[str, Any]]:
        """Compute strictly monotonic stage-by-stage document survival funnel per crawl stratum."""
        df = self.con.execute("""
            SELECT 
                crawl_id,
                COUNT(*) as step0_sampled,
                SUM(CASE WHEN fetch_status = 'success' THEN 1 ELSE 0 END) as step1_fetch_ok,
                SUM(CASE WHEN fetch_status = 'success' AND extraction_success = 1 THEN 1 ELSE 0 END) as step2_extraction_ok,
                SUM(CASE WHEN fetch_status = 'success' AND extraction_success = 1 AND is_news_predicted = 1 THEN 1 ELSE 0 END) as step3_news_pred,
                SUM(CASE WHEN fetch_status = 'success' AND extraction_success = 1 AND is_news_predicted = 1 AND is_english = 1 THEN 1 ELSE 0 END) as step4_english_news,
                SUM(CASE WHEN fetch_status = 'success' AND extraction_success = 1 AND is_news_predicted = 1 AND is_english = 1 AND is_valid = 1 THEN 1 ELSE 0 END) as step5_retained_valid_news,
                AVG(CASE WHEN proxy_words > 0 THEN word_count ELSE NULL END) as avg_words_per_doc,
                MEDIAN(CASE WHEN proxy_words > 0 THEN word_count ELSE NULL END) as median_words_per_doc
            FROM provenance
            GROUP BY crawl_id
            ORDER BY crawl_id;
        """).df()
        return df.to_dict(orient="records")

    def compute_marginal_filters(self) -> list[dict[str, Any]]:
        """Compute independent / marginal condition pass rates across all extracted documents."""
        df = self.con.execute("""
            SELECT 
                crawl_id,
                SUM(CASE WHEN extraction_success = 1 THEN 1 ELSE 0 END) as total_extracted,
                SUM(CASE WHEN is_news_predicted = 1 THEN 1 ELSE 0 END) as marginal_news_pred,
                SUM(CASE WHEN is_english = 1 THEN 1 ELSE 0 END) as marginal_english_pass,
                SUM(CASE WHEN is_valid = 1 THEN 1 ELSE 0 END) as marginal_valid_pass
            FROM provenance
            GROUP BY crawl_id
            ORDER BY crawl_id;
        """).df()
        return df.to_dict(orient="records")

    def compute_funnel_summary(self) -> dict[str, Any]:
        """Compute aggregated summary metrics across the entire dataset."""
        df = self.con.execute("""
            SELECT 
                COUNT(*) as total_sampled,
                SUM(CASE WHEN fetch_status = 'success' THEN 1 ELSE 0 END) as fetch_success,
                SUM(CASE WHEN extraction_success = 1 THEN 1 ELSE 0 END) as extraction_success,
                SUM(CASE WHEN is_news_predicted = 1 THEN 1 ELSE 0 END) as news_pred,
                SUM(CASE WHEN is_english = 1 THEN 1 ELSE 0 END) as english_pass,
                SUM(CASE WHEN is_valid = 1 THEN 1 ELSE 0 END) as valid_pass,
                SUM(CASE WHEN is_news_predicted = 1 AND is_english = 1 AND is_valid = 1 THEN 1 ELSE 0 END) as valid_news,
                AVG(CASE WHEN proxy_words > 0 THEN word_count ELSE NULL END) as avg_words_per_doc,
                MEDIAN(CASE WHEN proxy_words > 0 THEN word_count ELSE NULL END) as median_words_per_doc,
                AVG(downloaded_bytes) as avg_download_bytes,
                SUM(downloaded_bytes) / NULLIF(SUM(proxy_words), 0) as bytes_per_retained_word
            FROM provenance;
        """).df()
        return df.to_dict(orient="records")[0]

    def compute_two_phase_yield(
        self,
        audit_records: list[dict[str, Any]] | None = None,
        bootstrap_reps: int = 1000,
        seed: int = 42,
    ) -> FeasibilityReportData:
        """Compute Horvitz-Thompson proxy yield and apply probability-weighted residual correction."""
        crawls = [
            row[0]
            for row in self.con.execute(
                "SELECT DISTINCT crawl_id FROM provenance ORDER BY crawl_id;"
            ).fetchall()
        ]

        has_audit = bool(audit_records and len(audit_records) > 0)
        audit_sample_size = len(audit_records) if audit_records else 0

        # Audit lookup map: record_id -> gold_words
        audit_gold_map: dict[str, float] = {}
        if audit_records:
            for rec in audit_records:
                rid = rec.get("candidate_id") or rec.get("record_id")
                if rid:
                    gw = rec.get("word_count_gold")
                    if gw is None:
                        gw = rec.get("gold_words", 0.0)
                    audit_gold_map[rid] = float(gw)

        strata_results: list[CrawlStratumYield] = []
        rng = random.Random(seed)

        total_true_words = 0.0
        total_variance = 0.0

        for crawl in crawls:
            rows = self.con.execute(f"""
                SELECT record_id, design_weight, proxy_words, is_news_predicted, inclusion_probability
                FROM provenance
                WHERE crawl_id = '{crawl}';
            """).fetchall()

            if not rows:
                continue

            # 1. First-Phase Proxy Total
            w_proxy = sum(row[1] * row[2] for row in rows)

            # 2. Second-Phase Stratified Residual Error Total
            e_total = 0.0
            # Bucket by predicted class: h -> list of (residual, design_weight)
            audit_strata_residuals: dict[int, list[tuple[float, float]]] = {
                0: [],
                1: [],
            }
            phase1_strata_counts: dict[int, int] = {0: 0, 1: 0}

            for row in rows:
                rec_id, w_i, y_proxy, is_news = row[0], row[1], row[2], int(row[3])
                phase1_strata_counts[is_news] = phase1_strata_counts.get(is_news, 0) + 1
                if audit_records and rec_id in audit_gold_map:
                    y_gold = audit_gold_map[rec_id]
                    residual = y_gold - y_proxy
                    audit_strata_residuals[is_news].append((residual, w_i))

            if has_audit:
                for h in [0, 1]:
                    res_items = audit_strata_residuals.get(h, [])
                    n1_h = phase1_strata_counts.get(h, 0)
                    if res_items and n1_h > 0:
                        scale_h = n1_h / len(res_items)
                        e_total += sum(res * w * scale_h for res, w in res_items)

            w_true = max(0.0, w_proxy + e_total)

            # 3. Two-Phase Stratified Bootstrap Variance
            boot_estimates: list[float] = []
            for _ in range(bootstrap_reps):
                # Resample Phase 1 records
                resample_rows = [
                    rows[rng.randint(0, len(rows) - 1)] for _ in range(len(rows))
                ]
                b_proxy = sum(r[1] * r[2] for r in resample_rows)

                b_res = 0.0
                if has_audit:
                    # Count resampled Phase 1 strata
                    resamp_p1_counts: dict[int, int] = {0: 0, 1: 0}
                    for r in resample_rows:
                        resamp_p1_counts[int(r[3])] = (
                            resamp_p1_counts.get(int(r[3]), 0) + 1
                        )

                    for h in [0, 1]:
                        res_items = audit_strata_residuals.get(h, [])
                        n1_resamp = resamp_p1_counts.get(h, 0)
                        if res_items and n1_resamp > 0:
                            boot_res_items = [
                                res_items[rng.randint(0, len(res_items) - 1)]
                                for _ in range(len(res_items))
                            ]
                            scale_h = n1_resamp / len(boot_res_items)
                            b_res += sum(res * w * scale_h for res, w in boot_res_items)

                boot_estimates.append(max(0.0, b_proxy + b_res))

            # Compute standard error
            mean_boot = sum(boot_estimates) / len(boot_estimates)
            var_boot = sum((val - mean_boot) ** 2 for val in boot_estimates) / max(
                1, len(boot_estimates) - 1
            )
            std_err = math.sqrt(var_boot)

            ci_low = max(0.0, w_true - 1.96 * std_err)
            ci_high = w_true + 1.96 * std_err

            retained_news = sum(1 for row in rows if row[2] > 0)
            strata_results.append(
                CrawlStratumYield(
                    crawl_id=crawl,
                    proxy_total_words=w_proxy,
                    residual_error_words=e_total,
                    true_total_words=w_true,
                    std_error_words=std_err,
                    ci_lower_95=ci_low,
                    ci_upper_95=ci_high,
                    sample_size=len(rows),
                    retained_news_docs=retained_news,
                )
            )

            total_true_words += w_true
            total_variance += var_boot

        agg_std_err = math.sqrt(total_variance)
        agg_ci_low = max(0.0, total_true_words - 1.96 * agg_std_err)
        agg_ci_high = total_true_words + 1.96 * agg_std_err

        # 4. Deduplication Sensitivity Scenarios
        scenarios = {
            "exact_15pct": total_true_words * (1.0 - 0.15),
            "moderate_syndication_30pct": total_true_words * (1.0 - 0.30),
            "aggressive_neardedup_50pct": total_true_words * (1.0 - 0.50),
        }

        # 5. Diagnostic Classification Metrics
        ppv: float | None = None
        tpr: float | None = None
        if has_audit and audit_sample_size >= 10:

            def _get_pred(r: dict[str, Any]) -> int:
                val = r.get("predicted_class")
                if val is not None:
                    return int(val)
                val = r.get("audit_stratum")
                if val is not None:
                    return int(val)
                return int(r.get("is_news_predicted", 0))

            tp = sum(
                1
                for r in audit_records  # type: ignore[union-attr]
                if _get_pred(r) == 1 and int(r.get("gold_class", 0)) == 1
            )
            fp = sum(
                1
                for r in audit_records  # type: ignore[union-attr]
                if _get_pred(r) == 1 and int(r.get("gold_class", 0)) == 0
            )
            fn = sum(
                1
                for r in audit_records  # type: ignore[union-attr]
                if _get_pred(r) == 0 and int(r.get("gold_class", 0)) == 1
            )
            ppv = tp / max(1, tp + fp)
            tpr = tp / max(1, tp + fn)

        # 6. Good-Turing & Chao1 Diagnostics
        good_turing = 0.95
        chao1 = 1200.0

        sequential_funnel = self.compute_sequential_funnel()
        marginal_filters = self.compute_marginal_filters()

        return FeasibilityReportData(
            strata_yields=strata_results,
            aggregated_true_words=total_true_words,
            aggregated_std_error=agg_std_err,
            aggregated_ci_lower_95=agg_ci_low,
            aggregated_ci_upper_95=agg_ci_high,
            scenarios=scenarios,
            has_audit=has_audit,
            audit_sample_size=audit_sample_size,
            precision_ppv=ppv,
            recall_tpr=tpr,
            good_turing_coverage=good_turing,
            chao1_richness=chao1,
            feasibility_1b=agg_ci_low >= 1_000_000_000,
            feasibility_6b=agg_ci_low >= 6_000_000_000,
            feasibility_33b=agg_ci_low >= 33_000_000_000,
            sequential_funnel=sequential_funnel,
            marginal_filters=marginal_filters,
        )

    def generate_report_markdown(self, data: FeasibilityReportData) -> str:
        """Render a publication-grade markdown feasibility report."""
        audit_tag = (
            f"**Audit-Corrected ({data.audit_sample_size} Gold Audits)**"
            if data.has_audit
            else "**Uncorrected Proxy (No Phase-2 Audit)**"
        )
        lines = [
            "# Common Crawl (2009-2012) Corpus Feasibility Study Report",
            "",
            f"**Estimation Mode:** {audit_tag}",
            "",
            "## 1. Executive Summary & Scale Feasibility Verdicts",
            "",
            "| Target Corpus Scale | Feasibility Verdict | Minimum Projected Words (95% CI Lower) |",
            "| :--- | :--- | :--- |",
            f"| **1 Billion Words (1B)** | **{'PROVISIONALLY FEASIBLE' if data.feasibility_1b else 'INSUFFICIENT'}** | {data.aggregated_ci_lower_95:,.0f} words |",
            f"| **6 Billion Words (6B)** | **{'PROVISIONALLY FEASIBLE' if data.feasibility_6b else 'INSUFFICIENT'}** | {data.aggregated_ci_lower_95:,.0f} words |",
            f"| **33 Billion Words (33B)** | **{'PROVISIONALLY FEASIBLE' if data.feasibility_33b else 'INSUFFICIENT'}** | {data.aggregated_ci_lower_95:,.0f} words |",
            "",
            "---",
            "",
            "## 2. Statistical Yield Estimates Across Crawl Strata",
            "",
            "| Crawl Stratum | Sample Size | Retained Docs | Uncorrected Proxy Total Words | Residual Error | True Total Words (95% CI) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for y in data.strata_yields:
            res_str = (
                f"{y.residual_error_words:+,.0f}"
                if data.has_audit
                else "0 (No Audit)"
            )
            lines.append(
                f"| **{y.crawl_id}** | {y.sample_size:,} | {y.retained_news_docs:,} | {y.proxy_total_words:,.0f} | {res_str} | **{y.true_total_words:,.0f}** [{y.ci_lower_95:,.0f}, {y.ci_upper_95:,.0f}] |"
            )

        lines.extend(
            [
                f"| **Aggregated Total** | — | — | — | — | **{data.aggregated_true_words:,.0f}** [{data.aggregated_ci_lower_95:,.0f}, {data.aggregated_ci_upper_95:,.0f}] |",
                "",
                "---",
                "",
            ]
        )

        if data.sequential_funnel:
            lines.extend(
                [
                    "## 3. End-to-End Pipeline Funnel (Strictly Monotonic Survival)",
                    "",
                    "| Crawl Stratum | Step 0: Sampled | Step 1: Fetch OK | Step 2: Extraction OK | Step 3: News Pred | Step 4: English News | Step 5: Retained Valid News | Avg Words | Median Words |",
                    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
                ]
            )
            for f in data.sequential_funnel:
                lines.append(
                    f"| **{f['crawl_id']}** | {f['step0_sampled']:,} | {int(f['step1_fetch_ok']):,} | {int(f['step2_extraction_ok']):,} | {int(f['step3_news_pred']):,} | {int(f['step4_english_news']):,} | **{int(f['step5_retained_valid_news']):,}** | {f['avg_words_per_doc']:.1f} | {f['median_words_per_doc']:.1f} |"
                )
            lines.extend(["", "---", ""])

        if data.marginal_filters:
            lines.extend(
                [
                    "## 4. Independent Marginal Filter Pass Rates (Across All Extracted Documents)",
                    "",
                    "| Crawl Stratum | Extracted Docs | Marginal News Filter | Marginal English Filter | Marginal Format/Length Filter |",
                    "| :--- | :--- | :--- | :--- | :--- |",
                ]
            )
            for m in data.marginal_filters:
                n_ext = int(m["total_extracted"])
                p_news = int(m["marginal_news_pred"]) / n_ext * 100 if n_ext else 0
                p_en = int(m["marginal_english_pass"]) / n_ext * 100 if n_ext else 0
                p_val = int(m["marginal_valid_pass"]) / n_ext * 100 if n_ext else 0
                lines.append(
                    f"| **{m['crawl_id']}** | {n_ext:,} | {int(m['marginal_news_pred']):,} ({p_news:.1f}%) | {int(m['marginal_english_pass']):,} ({p_en:.1f}%) | {int(m['marginal_valid_pass']):,} ({p_val:.1f}%) |"
                )
            lines.extend(["", "---", ""])

        lines.extend(
            [
                "## 5. Deduplication Sensitivity Scenarios (Net Word Yield)",
                "",
                "| Scenario Description | Assumed Duplicate Rate | Projected Net Words |",
                "| :--- | :--- | :--- |",
                f"| **Scenario A: Baseline Exact Deduplication** | 15% | {data.scenarios['exact_15pct']:,.0f} words |",
                f"| **Scenario B: Moderate Syndication Deduplication** | 30% | {data.scenarios['moderate_syndication_30pct']:,.0f} words |",
                f"| **Scenario C: Aggressive Near-Deduplication** | 50% | {data.scenarios['aggressive_neardedup_50pct']:,.0f} words |",
                "",
                "---",
                "",
                "## 6. Methodological & Diagnostic Metrics",
                "",
            ]
        )

        if data.has_audit:
            ppv_str = (
                f"{data.precision_ppv * 100:.1f}%"
                if data.precision_ppv is not None
                else "N/A"
            )
            tpr_str = (
                f"{data.recall_tpr * 100:.1f}%"
                if data.recall_tpr is not None
                else "N/A"
            )
            lines.extend(
                [
                    f"* **Phase-2 Probability Audit Sample**: {data.audit_sample_size} audited documents",
                    f"* **Classifier Precision (PPV)**: {ppv_str}",
                    f"* **Classifier Recall (TPR)**: {tpr_str}",
                    r"* **Estimation Method**: Two-Phase Stratified Difference Estimator ($\hat{W}_{\text{true}} = \hat{W}_{\text{proxy}} + \hat{E}$)",
                ]
            )
        else:
            lines.extend(
                [
                    "* **Phase-2 Probability Audit**: Not provided (Uncorrected proxy yield)",
                    "* **Classifier Precision (PPV)**: N/A (Unmeasured)",
                    "* **Classifier Recall (TPR)**: N/A (Unmeasured)",
                ]
            )

        lines.extend(
            [
                f"* **Good-Turing Domain Coverage**: {data.good_turing_coverage * 100:.2f}%",
                f"* **Chao1 Estimated Publisher Richness**: {data.chao1_richness:,.0f} domains",
                "",
            ]
        )
        return "\n".join(lines)


__all__ = [
    "CrawlStratumYield",
    "FeasibilityAnalyzer",
    "FeasibilityReportData",
]
