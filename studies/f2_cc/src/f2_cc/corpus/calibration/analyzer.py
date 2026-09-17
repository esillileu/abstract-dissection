"""Calibration and pre-fetch feasibility analyzer coordinator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from .models import (
    PostFetchOperatingPoint,
    ProductionPipelineRecommendation,
    RuleAblationResult,
)
from .postfetch import evaluate_cross_validated_postfetch, evaluate_postfetch_sweep
from .recommendation import evaluate_production_recommendations
from .reporting import generate_report_markdown
from .rules import (
    ALL_BINARY_EXT,
    DISQUALIFIED_PATH,
    NON_NEWS_PATTERNS,
    NON_PDF_MEDIA_EXT,
    PDF_EXT,
    evaluate_rule_ablation,
    get_stratum_id,
)


class CalibrationAndPreFetchAnalyzer:
    """Performs offline calibration and pre-fetch filtering feasibility on fixed 10k/400 sample."""

    NON_PDF_MEDIA_EXT = NON_PDF_MEDIA_EXT
    PDF_EXT = PDF_EXT
    ALL_BINARY_EXT = ALL_BINARY_EXT
    DISQUALIFIED_PATH = DISQUALIFIED_PATH
    NON_NEWS_PATTERNS = NON_NEWS_PATTERNS
    get_stratum_id = staticmethod(get_stratum_id)

    def __init__(
        self,
        provenance_path: Path,
        audit_records: list[dict[str, Any]],
    ) -> None:
        self.provenance_path = provenance_path
        self.raw_audit_records = audit_records
        self.con = duckdb.connect(":memory:")
        self._load_data()
        self._prepare_weights()

    def _load_data(self) -> None:
        posix_path = self.provenance_path.as_posix()
        if posix_path.endswith(".parquet"):
            self.con.execute(
                f"CREATE TABLE provenance AS SELECT * FROM read_parquet('{posix_path}');"
            )
        else:
            self.con.execute(
                f"CREATE TABLE provenance AS SELECT * FROM read_json_auto('{posix_path}');"
            )

        df = self.con.execute("SELECT * FROM provenance").df()
        self.all_records: list[dict[str, Any]] = df.to_dict(orient="records")

    def _prepare_weights(self) -> None:
        p1_counts: dict[str, int] = {}
        for r in self.all_records:
            pref = r.get("prefilter_status", "pass")
            sid = self.get_stratum_id(
                r["crawl_id"], pref, r.get("is_news_predicted", 0)
            )
            p1_counts[sid] = p1_counts.get(sid, 0) + 1

        aud_counts: dict[str, int] = {}
        for a in self.raw_audit_records:
            rid = a.get("candidate_id") or a.get("record_id")
            rec = next(
                (r for r in self.all_records if str(r["record_id"]) == str(rid)),
                None,
            )
            if rec:
                pref = rec.get("prefilter_status", "pass")
                raw_strat = a.get("design_stratum") or a.get("audit_stratum")
                if str(raw_strat).startswith("S"):
                    sid = str(raw_strat)
                else:
                    sid = self.get_stratum_id(
                        rec["crawl_id"], pref, rec.get("is_news_predicted", 0)
                    )
                aud_counts[sid] = aud_counts.get(sid, 0) + 1

        self.audited_items: list[dict[str, Any]] = []
        for a in self.raw_audit_records:
            rid = a.get("candidate_id") or a.get("record_id")
            rec = next(
                (r for r in self.all_records if str(r["record_id"]) == str(rid)),
                None,
            )
            if rec:
                item = dict(rec)
                item["gold_class"] = int(a.get("gold_class", 0))
                item["word_count_gold"] = float(
                    a.get("word_count_gold", a.get("gold_words", 0.0))
                )
                pref = rec.get("prefilter_status", "pass")
                raw_strat = a.get("design_stratum") or a.get("audit_stratum")
                if str(raw_strat).startswith("S"):
                    sid = str(raw_strat)
                else:
                    sid = self.get_stratum_id(
                        rec["crawl_id"], pref, rec.get("is_news_predicted", 0)
                    )
                item["design_stratum"] = sid
                item["audit_stratum"] = int(
                    a.get(
                        "predicted_class",
                        1
                        if "pos" in sid.lower() or sid in ["S1", "S3", "S5", "S7"]
                        else 0,
                    )
                )
                w2 = (
                    p1_counts[sid] / aud_counts[sid]
                    if aud_counts.get(sid, 0) > 0
                    else 1.0
                )
                item["total_weight"] = float(rec["design_weight"]) * w2
                self.audited_items.append(item)

    def evaluate_rule_ablation(self) -> list[RuleAblationResult]:
        """Perform fine-grained ablation of pre-fetch rules on full 10k population."""
        return evaluate_rule_ablation(self.all_records)

    def evaluate_postfetch_sweep(self) -> list[PostFetchOperatingPoint]:
        """Sweep post-fetch news_score threshold across full audited sample."""
        return evaluate_postfetch_sweep(self.audited_items)

    def evaluate_cross_validated_postfetch(
        self, target_recalls: list[float] | None = None
    ) -> list[dict[str, Any]]:
        """Evaluate out-of-fold cross-validated generalization at target operating points."""
        return evaluate_cross_validated_postfetch(self.audited_items, target_recalls)

    def evaluate_production_recommendations(
        self,
    ) -> list[ProductionPipelineRecommendation]:
        """Direct joint end-to-end evaluation on gold sample (without marginal multiplication)."""
        return evaluate_production_recommendations(self.all_records, self.audited_items)

    def generate_report_markdown(self) -> str:
        """Generate comprehensive publication-grade markdown analysis."""
        ablation_results = self.evaluate_rule_ablation()
        post_sweep = self.evaluate_postfetch_sweep()
        post_cv = self.evaluate_cross_validated_postfetch()
        prod_recs = self.evaluate_production_recommendations()
        return generate_report_markdown(
            ablation_results, post_sweep, post_cv, prod_recs
        )


__all__ = ["CalibrationAndPreFetchAnalyzer"]
