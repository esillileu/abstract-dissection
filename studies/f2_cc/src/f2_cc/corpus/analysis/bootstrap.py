"""Two-stage cluster & subsampling bootstrap estimator across 8 strata."""

from __future__ import annotations

import math
import random
from typing import Any

import duckdb

from .models import CrawlStratumYield
from .strata import get_stratum_id


def evaluate_audit_bootstrap(
    con: duckdb.DuckDBPyConnection,
    audit_records: list[dict[str, Any]] | None,
    bootstrap_reps: int = 1000,
    seed: int = 42,
) -> tuple[list[CrawlStratumYield], float, float, float, float]:
    """Core internal engine implementing 3-stage cluster & subsampling bootstrap across 8 strata."""
    crawls = [
        row[0]
        for row in con.execute(
            "SELECT DISTINCT crawl_id FROM provenance ORDER BY crawl_id;"
        ).fetchall()
    ]

    has_audit = bool(audit_records and len(audit_records) > 0)
    audit_gold_map: dict[str, float] = {}
    audit_gold_class: dict[str, int] = {}
    if audit_records:
        for rec in audit_records:
            rid = rec.get("candidate_id") or rec.get("record_id")
            if rid:
                gw = rec.get("word_count_gold")
                if gw is None:
                    gw = rec.get("gold_words", 0.0)
                audit_gold_map[rid] = float(gw)
                gc = rec.get("gold_class")
                if gc is not None:
                    audit_gold_class[rid] = int(gc)

    strata_results: list[CrawlStratumYield] = []
    rng = random.Random(seed)
    total_true_words = 0.0
    total_variance = 0.0

    # Check table columns
    table_cols = [
        row[1] for row in con.execute("PRAGMA table_info('provenance');").fetchall()
    ]
    has_block_idx = "block_index" in table_cols
    has_pref = "prefilter_status" in table_cols
    has_fetch_prob = "fetch_probability" in table_cols

    for crawl in crawls:
        col_block = "block_index" if has_block_idx else "0 as block_index"
        col_pref = "prefilter_status" if has_pref else "'pass' as prefilter_status"
        col_fprob = (
            "fetch_probability" if has_fetch_prob else "1.0 as fetch_probability"
        )

        query = f"""
            SELECT 
                record_id, 
                design_weight, 
                proxy_words, 
                is_news_predicted, 
                inclusion_probability,
                {col_block},
                {col_pref},
                {col_fprob}
            FROM provenance
            WHERE crawl_id = '{crawl}';
        """
        rows = con.execute(query).fetchall()

        if not rows:
            continue

        # Point estimate computation
        w_proxy = sum(row[1] * row[2] for row in rows)

        # Map audit residuals to design strata (S1 to S8)
        audit_strata_residuals: dict[str, list[tuple[float, float, str]]] = {
            f"S{i}": [] for i in range(1, 9)
        }
        phase1_strata_counts: dict[str, int] = {f"S{i}": 0 for i in range(1, 9)}

        tp_w = fp_w = fn_w = 0.0

        for row in rows:
            rec_id = row[0]
            w_i = float(row[1])
            y_proxy = float(row[2])
            is_news = int(row[3])
            pref_stat = str(row[6])

            sid = get_stratum_id(crawl, pref_stat, is_news)
            phase1_strata_counts[sid] = phase1_strata_counts.get(sid, 0) + 1

            if audit_records and rec_id in audit_gold_map:
                y_gold = audit_gold_map[rec_id]
                residual = y_gold - y_proxy
                audit_strata_residuals[sid].append((residual, w_i, rec_id))

                g_cls = audit_gold_class.get(rec_id, 1 if y_gold > 0 else 0)
                if is_news == 1:
                    if g_cls == 1:
                        tp_w += w_i
                    else:
                        fp_w += w_i
                else:
                    if g_cls == 1:
                        fn_w += w_i

        e_total = 0.0
        if has_audit:
            for sid in sorted(phase1_strata_counts.keys()):
                res_items = audit_strata_residuals.get(sid, [])
                n1_h = phase1_strata_counts.get(sid, 0)
                if res_items and n1_h > 0:
                    scale_h = n1_h / len(res_items)
                    e_total += sum(res * w * scale_h for res, w, _ in res_items)

        w_true = max(0.0, w_proxy + e_total)

        # Two-Stage Cluster Bootstrap with Reject-Exploration Subsampling
        blocks_dict: dict[int, list[Any]] = {}
        for row in rows:
            b_idx = int(row[5])
            blocks_dict.setdefault(b_idx, []).append(row)

        block_keys = list(blocks_dict.keys())
        num_blocks = len(block_keys)

        boot_estimates: list[float] = []
        for _ in range(bootstrap_reps):
            # Stage 1: Resample blocks with replacement
            resample_blocks = [
                blocks_dict[block_keys[rng.randint(0, num_blocks - 1)]]
                for _ in range(num_blocks)
            ]
            # Stage 2: Resample records within selected blocks
            resample_rows = []
            for blk in resample_blocks:
                m_k = len(blk)
                for _ in range(m_k):
                    resample_rows.append(blk[rng.randint(0, m_k - 1)])

            # Stage 3: Apply reject-exploration subsampling multiplier
            b_proxy = 0.0
            resamp_p1_counts: dict[str, int] = {f"S{i}": 0 for i in range(1, 9)}
            rec_mult_map: dict[str, float] = {}

            for r in resample_rows:
                rec_id = r[0]
                base_w = float(r[1])
                y_p = float(r[2])
                is_n = int(r[3])
                p_stat = str(r[6])
                f_prob = float(r[7])

                # Multiplier for reject exploration stream
                if p_stat == "reject" and f_prob < 0.99:
                    var_mult = (1.0 - f_prob) / max(1e-6, f_prob)
                    shape = 1.0 / var_mult
                    scale = var_mult
                    mult = rng.gammavariate(shape, scale)
                else:
                    mult = 1.0

                rec_mult_map[rec_id] = mult
                adj_w = base_w * mult
                b_proxy += adj_w * y_p

                sid = get_stratum_id(crawl, p_stat, is_n)
                resamp_p1_counts[sid] = resamp_p1_counts.get(sid, 0) + 1

            b_res = 0.0
            if has_audit:
                for sid in sorted(resamp_p1_counts.keys()):
                    res_items = audit_strata_residuals.get(sid, [])
                    n1_resamp = resamp_p1_counts.get(sid, 0)
                    if res_items and n1_resamp > 0:
                        boot_res_items = [
                            res_items[rng.randint(0, len(res_items) - 1)]
                            for _ in range(len(res_items))
                        ]
                        scale_h = n1_resamp / len(boot_res_items)
                        b_res += sum(
                            res * w * rec_mult_map.get(rid, 1.0) * scale_h
                            for res, w, rid in boot_res_items
                        )

            boot_estimates.append(max(0.0, b_proxy + b_res))

        mean_boot = sum(boot_estimates) / len(boot_estimates)
        var_boot = sum((val - mean_boot) ** 2 for val in boot_estimates) / max(
            1, len(boot_estimates) - 1
        )
        std_err = math.sqrt(var_boot)
        ci_low = max(0.0, w_true - 1.96 * std_err)
        ci_high = w_true + 1.96 * std_err

        crawl_ppv = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else None
        crawl_tpr = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else None

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
                weighted_ppv=crawl_ppv,
                weighted_tpr=crawl_tpr,
            )
        )
        total_true_words += w_true
        total_variance += var_boot

    agg_std_err = math.sqrt(total_variance)
    agg_ci_low = max(0.0, total_true_words - 1.96 * agg_std_err)
    agg_ci_high = total_true_words + 1.96 * agg_std_err

    return strata_results, total_true_words, agg_std_err, agg_ci_low, agg_ci_high
