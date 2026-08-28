"""Corpus state repository implementing atomic transitions and queries for PostgreSQL."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq

from ..discovery import CandidateRecord
from ..pipeline import ProcessedDocumentResult


class CorpusStateRepository:
    """PostgreSQL-backed operational state repository for the F2 corpus pipeline."""

    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        self.conn = conn

    def create_run(
        self,
        run_id: str,
        run_type: str,
        crawl_ids: list[str],
        sample_size: int,
        seed: int,
        bandwidth_mbps: float,
        concurrency: int,
        output_dir: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pipeline_runs (
                    run_id, run_type, crawl_ids, sample_size, seed,
                    bandwidth_mbps, concurrency, status, output_dir, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'running', %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    status = 'running',
                    finished_at = NULL,
                    metadata = EXCLUDED.metadata;
                """,
                (
                    run_id,
                    run_type,
                    crawl_ids,
                    sample_size,
                    seed,
                    bandwidth_mbps,
                    concurrency,
                    output_dir,
                    json.dumps(metadata or {}),
                ),
            )
        self.conn.commit()
        return run_id

    def update_run_status(self, run_id: str, status: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE pipeline_runs
                SET status = %s, finished_at = %s
                WHERE run_id = %s;
                """,
                (status, datetime.now(UTC), run_id),
            )
        self.conn.commit()

    def insert_candidates(self, run_id: str, candidates: list[CandidateRecord]) -> int:
        if not candidates:
            return 0
        count = 0
        with self.conn.cursor() as cur:
            for c in candidates:
                cand_id = c.record_id()
                cur.execute(
                    """
                    INSERT INTO candidate_records (
                        candidate_id, run_id, crawl_id, url, url_timestamp,
                        arc_filename, arc_offset, arc_length, arc_digest,
                        source_type, stratum, inclusion_probability, design_weight,
                        block_index, record_index_in_block, block_total_records,
                        prefilter_status, prefilter_rule, fetch_probability, is_selected_for_fetch
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, candidate_id) DO NOTHING;
                    """,
                    (
                        cand_id,
                        run_id,
                        c.crawl_id,
                        c.url,
                        c.timestamp,
                        c.filename,
                        c.offset,
                        c.length,
                        c.digest,
                        c.source_type,
                        c.stratum,
                        c.inclusion_probability,
                        c.design_weight,
                        c.block_index,
                        c.record_index_in_block,
                        c.block_total_records,
                        c.prefilter_status,
                        c.prefilter_rule,
                        c.fetch_probability,
                        c.is_selected_for_fetch,
                    ),
                )
                count += cur.rowcount
        self.conn.commit()
        return count

    def get_completed_candidate_ids(self, run_id: str) -> set[str]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT candidate_id FROM processing_results
                WHERE run_id = %s;
                """,
                (run_id,),
            )
            return {row[0] for row in cur.fetchall()}

    def record_processing_result(
        self,
        run_id: str,
        result: ProcessedDocumentResult,
        clean_text_sha256: str | None = None,
        shard_path: str | None = None,
        prefilter_status: str = "pass",
        is_reject_exploration: bool = False,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO processing_results (
                    candidate_id, run_id, fetch_status, http_status,
                    downloaded_bytes, extraction_success, news_score,
                    is_news_predicted, is_english, is_valid, rejection_reason,
                    word_count, word_count_proxy, clean_text_sha256, shard_path,
                    prefilter_status, is_reject_exploration, diagnostics
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, candidate_id) DO UPDATE SET
                    fetch_status = EXCLUDED.fetch_status,
                    http_status = EXCLUDED.http_status,
                    downloaded_bytes = EXCLUDED.downloaded_bytes,
                    extraction_success = EXCLUDED.extraction_success,
                    news_score = EXCLUDED.news_score,
                    is_news_predicted = EXCLUDED.is_news_predicted,
                    is_english = EXCLUDED.is_english,
                    is_valid = EXCLUDED.is_valid,
                    rejection_reason = EXCLUDED.rejection_reason,
                    word_count = EXCLUDED.word_count,
                    word_count_proxy = EXCLUDED.word_count_proxy,
                    clean_text_sha256 = EXCLUDED.clean_text_sha256,
                    shard_path = EXCLUDED.shard_path,
                    prefilter_status = EXCLUDED.prefilter_status,
                    is_reject_exploration = EXCLUDED.is_reject_exploration,
                    diagnostics = EXCLUDED.diagnostics,
                    processed_at = NOW();
                """,
                (
                    result.record_id,
                    run_id,
                    result.fetch_status,
                    result.http_status,
                    result.downloaded_bytes,
                    result.extraction_success,
                    result.news_score,
                    result.is_news_predicted,
                    result.is_english,
                    result.is_valid,
                    result.rejection_reason,
                    result.word_count,
                    result.proxy_words,
                    clean_text_sha256,
                    shard_path,
                    prefilter_status,
                    is_reject_exploration,
                    json.dumps(result.diagnostics),
                ),
            )
        self.conn.commit()

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
        self.conn.commit()
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
        self.conn.commit()

    def get_provenance_records(self, run_id: str) -> list[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    c.candidate_id AS record_id,
                    c.crawl_id,
                    c.url,
                    c.inclusion_probability,
                    c.design_weight,
                    c.block_index,
                    c.record_index_in_block,
                    c.block_total_records,
                    c.prefilter_status,
                    c.prefilter_rule,
                    c.fetch_probability,
                    c.is_selected_for_fetch,
                    r.fetch_status,
                    r.http_status,
                    r.downloaded_bytes,
                    r.extraction_success,
                    r.news_score,
                    r.is_news_predicted,
                    r.is_english,
                    r.is_valid,
                    r.rejection_reason,
                    r.word_count,
                    r.word_count_proxy AS proxy_words,
                    r.clean_text_sha256,
                    r.shard_path,
                    r.diagnostics,
                    COALESCE(a.is_audited, FALSE) AS is_audited,
                    a.gold_class,
                    a.word_count_gold,
                    a.word_residual,
                    a.audit_inclusion_probability,
                    a.audit_design_weight,
                    a.design_stratum
                FROM candidate_records c
                JOIN processing_results r ON c.run_id = r.run_id AND c.candidate_id = r.candidate_id
                LEFT JOIN audit_assignments a ON c.run_id = a.run_id AND c.candidate_id = a.candidate_id
                WHERE c.run_id = %s
                ORDER BY c.block_index, c.record_index_in_block;
                """,
                (run_id,),
            )
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return [dict(zip(cols, row, strict=False)) for row in rows]

    def export_provenance_to_jsonl(self, run_id: str, output_path: Path) -> int:
        records = self.get_provenance_records(run_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, default=str) + "\n")
        return len(records)

    def export_provenance_to_parquet(self, run_id: str, output_path: Path) -> int:
        records = self.get_provenance_records(run_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not records:
            return 0

        clean_records = []
        for r in records:
            cr = dict(r)
            cr["is_news_predicted"] = int(cr["is_news_predicted"])
            cr["is_english"] = int(cr["is_english"])
            cr["is_valid"] = int(cr["is_valid"])
            cr["extraction_success"] = int(cr["extraction_success"])
            cr["is_selected_for_fetch"] = int(
                bool(cr.get("is_selected_for_fetch", True))
            )
            cr["is_audited"] = int(bool(cr["is_audited"]))
            cr["diagnostics"] = json.dumps(cr["diagnostics"] or {})
            clean_records.append(cr)

        table = pa.Table.from_pylist(clean_records)
        pq.write_table(table, output_path.as_posix(), compression="zstd")
        return len(records)


__all__ = ["CorpusStateRepository"]
