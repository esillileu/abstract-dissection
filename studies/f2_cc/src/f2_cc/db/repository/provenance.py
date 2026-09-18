"""Provenance records extraction and serialization to JSONL and Parquet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .base import BaseRepository


class ProvenanceRepositoryMixin(BaseRepository):
    """Operations for querying provenance records and exporting to Parquet/JSONL."""

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
