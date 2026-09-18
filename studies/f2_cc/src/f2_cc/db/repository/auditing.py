"""Audit assignments, gold label recording, and audit record retrieval."""

from __future__ import annotations

from typing import Any

from .base import BaseRepository


class AuditingRepositoryMixin(BaseRepository):
    """Operations for recording audit waves, gold labels, and audit queries."""

    def insert_audit_assignments(
        self, run_id: str, assignments: list[dict[str, Any]]
    ) -> int:
        if not assignments:
            return 0
        count = 0
        with self.conn.cursor() as cur:
            for item in assignments:
                cand_id = item.get("candidate_id") or item.get("record_id")
                audit_id = f"{run_id}:{cand_id}"
                design_strat = item.get(
                    "design_stratum",
                    "S1" if item.get("predicted_class", 1) == 1 else "S2",
                )
                audit_strat = int(
                    item.get(
                        "audit_stratum",
                        1 if item.get("is_news_predicted", 0) == 1 else 0,
                    )
                )
                cur.execute(
                    """
                    INSERT INTO audit_assignments (
                        audit_id, run_id, candidate_id, audit_stratum, priority_order,
                        wave, audit_inclusion_probability, audit_design_weight, design_stratum
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, candidate_id) DO NOTHING;
                    """,
                    (
                        audit_id,
                        run_id,
                        cand_id,
                        audit_strat,
                        item["priority_order"],
                        item.get("wave", 1),
                        item["audit_inclusion_prob_cond"],
                        item["audit_weight_cond"],
                        design_strat,
                    ),
                )
                count += cur.rowcount
        self._commit()
        return count

    def get_audit_assignments(self, run_id: str) -> list[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    a.audit_id,
                    a.run_id,
                    a.candidate_id,
                    a.audit_stratum,
                    a.priority_order,
                    a.wave,
                    a.audit_inclusion_probability,
                    a.audit_design_weight,
                    a.design_stratum,
                    a.is_audited,
                    a.gold_class,
                    a.word_count_gold,
                    a.word_residual,
                    a.auditor_id,
                    a.notes,
                    c.crawl_id,
                    c.url,
                    c.prefilter_status,
                    c.fetch_probability,
                    c.inclusion_probability AS first_stage_pi,
                    c.design_weight AS first_stage_weight,
                    r.news_score,
                    r.is_news_predicted,
                    r.is_english,
                    r.is_valid,
                    r.word_count,
                    r.word_count_proxy,
                    r.shard_path,
                    r.clean_text_sha256,
                    r.diagnostics
                FROM audit_assignments a
                JOIN candidate_records c ON a.run_id = c.run_id AND a.candidate_id = c.candidate_id
                JOIN processing_results r ON a.run_id = r.run_id AND a.candidate_id = r.candidate_id
                WHERE a.run_id = %s
                ORDER BY a.priority_order;
                """,
                (run_id,),
            )
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return [dict(zip(cols, row, strict=False)) for row in rows]

    def record_audit_gold_label(
        self,
        run_id: str,
        candidate_id: str,
        gold_class: int,
        word_count_gold: int,
        auditor_id: str = "human",
        notes: str | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT word_count_proxy FROM processing_results WHERE run_id = %s AND candidate_id = %s;",
                (run_id, candidate_id),
            )
            row = cur.fetchone()
            y_proxy = row[0] if row else 0
            residual = word_count_gold - y_proxy

            cur.execute(
                """
                UPDATE audit_assignments SET
                    is_audited = TRUE,
                    gold_class = %s,
                    word_count_gold = %s,
                    word_residual = %s,
                    audited_at = NOW(),
                    auditor_id = %s,
                    notes = %s
                WHERE run_id = %s AND candidate_id = %s;
                """,
                (
                    gold_class,
                    word_count_gold,
                    residual,
                    auditor_id,
                    notes,
                    run_id,
                    candidate_id,
                ),
            )
        self._commit()

    def get_profile_audit_records(self, run_id: str) -> list[dict[str, Any]]:
        """Return the gold audit projection for a DB-backed analysis run."""
        # Note: get_provenance_records is provided by ProvenanceRepositoryMixin
        return [
            row
            for row in self.get_provenance_records(run_id)  # type: ignore[attr-defined]
            if row.get("is_audited") and row.get("gold_class") is not None
        ]
