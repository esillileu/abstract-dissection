"""Database query and inspection for canonical Common Crawl runs."""

from __future__ import annotations

from typing import Any

from ..contracts import CANONICAL_PROFILES


def inspect_run(conn: Any, profile: str) -> dict[str, Any]:
    run_id = CANONICAL_PROFILES[profile]
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM pipeline_runs WHERE run_id=%s", (run_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"missing canonical run {run_id}")
        run = dict(zip((item.name for item in cur.description), row, strict=True))
        cur.execute(
            """SELECT c.candidate_id,c.crawl_id,c.url,c.arc_filename,c.arc_offset,
                       c.arc_length,c.inclusion_probability,c.design_weight,c.block_index,
                       c.record_index_in_block,r.clean_text_sha256,r.word_count
                       FROM candidate_records c JOIN processing_results r USING(run_id,candidate_id)
                       WHERE c.run_id=%s AND r.clean_text_sha256 IS NOT NULL
                       ORDER BY array_position(%s::text[],c.crawl_id),c.block_index,
                                c.record_index_in_block,c.candidate_id""",
            (run_id, run["crawl_ids"]),
        )
        columns = [item.name for item in cur.description]
        documents = [dict(zip(columns, item, strict=True)) for item in cur.fetchall()]
        cur.execute(
            "SELECT stage_role FROM stage_lineage WHERE source_run_id=%s", (run_id,)
        )
        lineage = [item[0] for item in cur.fetchall()]
    return {"profile": profile, "run": run, "documents": documents, "lineage": lineage}


__all__ = ["inspect_run"]
