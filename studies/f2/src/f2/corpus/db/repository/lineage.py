"""Recursive DAG lineage traversal operations for corpus shards."""

from __future__ import annotations

from typing import Any

from .base import BaseCorpusRepository


class LineageRepositoryMixin(BaseCorpusRepository):
    """Repository operations for recursive DAG lineage traversal."""

    def get_reverse_lineage(
        self,
        target_corpus_version: str,
        shard_index: int | None = None,
    ) -> list[dict[str, Any]]:
        """Traverse backwards through arbitrary-depth processing DAG to upstream raw sources."""
        shard_clause = "AND cs.shard_index = %s" if shard_index is not None else ""
        params: list[Any] = [target_corpus_version]
        if shard_index is not None:
            params.append(shard_index)

        query = f"""
        WITH RECURSIVE reverse_dag AS (
            SELECT 
                cs.resource_version_id AS target_corpus_version,
                cs.shard_index,
                a.artifact_id AS current_artifact_id,
                a.stage AS current_stage,
                a.s3_uri AS current_s3_uri,
                a.sha256 AS current_sha256,
                CAST(NULL AS VARCHAR(64)) AS via_run_id,
                CAST(NULL AS VARCHAR(64)) AS via_recipe,
                CAST(NULL AS VARCHAR(64)) AS via_code_version,
                0 AS depth,
                ARRAY[a.artifact_id]::varchar[] AS path
            FROM corpus_shards cs
            JOIN artifacts a ON a.artifact_id = cs.artifact_id
            WHERE cs.resource_version_id = %s
              {shard_clause}

            UNION ALL

            SELECT 
                rd.target_corpus_version,
                rd.shard_index,
                a_up.artifact_id AS current_artifact_id,
                a_up.stage AS current_stage,
                a_up.s3_uri AS current_s3_uri,
                a_up.sha256 AS current_sha256,
                pr.run_id AS via_run_id,
                pr.recipe_name AS via_recipe,
                pr.code_version AS via_code_version,
                rd.depth + 1 AS depth,
                rd.path || a_up.artifact_id
            FROM reverse_dag rd
            JOIN processing_run_outputs pro ON pro.artifact_id = rd.current_artifact_id
            JOIN processing_runs pr ON pr.run_id = pro.run_id
            JOIN processing_run_inputs pri ON pri.run_id = pr.run_id
            JOIN artifacts a_up ON a_up.artifact_id = pri.artifact_id
            WHERE NOT (a_up.artifact_id = ANY(rd.path))
        )
        SELECT 
            r.target_corpus_version,
            r.shard_index,
            r.depth AS hops_from_canonical,
            r.current_stage,
            r.current_artifact_id,
            r.current_s3_uri,
            r.via_recipe,
            r.via_run_id,
            r.via_code_version AS step_git_commit,
            rv_raw.version_label AS origin_release_label,
            res_src.name AS origin_source_name,
            rs_raw.license AS origin_license,
            acq.run_id AS origin_acquisition_run,
            acq.acquisition_method
        FROM reverse_dag r
        LEFT JOIN artifacts a_raw ON a_raw.artifact_id = r.current_artifact_id AND a_raw.stage = 'raw'
        LEFT JOIN catalog.resource_versions rv_raw ON rv_raw.resource_version_id = a_raw.resource_version_id
        LEFT JOIN catalog.resources res_src ON res_src.resource_id = rv_raw.resource_id
        LEFT JOIN catalog.resource_sources rs_raw ON rs_raw.resource_id = res_src.resource_id AND rs_raw.is_preferred = TRUE
        LEFT JOIN acquisition_runs acq ON acq.run_id = a_raw.acquisition_run_id
        ORDER BY r.shard_index, r.depth DESC;
        """
        with self.conn.cursor() as cur:
            cur.execute(query, tuple(params))
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]

    def get_forward_lineage(
        self,
        source_resource_version_id: str,
    ) -> list[dict[str, Any]]:
        """Traverse forward through processing DAG from a raw source release to canonical shards."""
        query = """
        WITH RECURSIVE forward_dag AS (
            SELECT 
                a.artifact_id AS current_artifact_id,
                a.stage AS current_stage,
                a.s3_uri AS current_s3_uri,
                CAST(NULL AS VARCHAR(64)) AS via_run_id,
                CAST(NULL AS VARCHAR(64)) AS via_recipe,
                0 AS depth,
                ARRAY[a.artifact_id]::varchar[] AS path
            FROM artifacts a
            WHERE a.resource_version_id = %s AND a.stage = 'raw'

            UNION ALL

            SELECT 
                a_down.artifact_id AS current_artifact_id,
                a_down.stage AS current_stage,
                a_down.s3_uri AS current_s3_uri,
                pr.run_id AS via_run_id,
                pr.recipe_name AS via_recipe,
                fd.depth + 1 AS depth,
                fd.path || a_down.artifact_id
            FROM forward_dag fd
            JOIN processing_run_inputs pri ON pri.artifact_id = fd.current_artifact_id
            JOIN processing_runs pr ON pr.run_id = pri.run_id
            JOIN processing_run_outputs pro ON pro.run_id = pr.run_id
            JOIN artifacts a_down ON a_down.artifact_id = pro.artifact_id
            WHERE NOT (a_down.artifact_id = ANY(fd.path))
        )
        SELECT 
            f.depth AS hop_depth,
            f.current_stage,
            f.current_artifact_id,
            f.current_s3_uri,
            f.via_recipe,
            f.via_run_id,
            cs.resource_version_id AS canonical_corpus_version,
            cs.shard_index,
            cs.word_count AS shard_words
        FROM forward_dag f
        LEFT JOIN corpus_shards cs ON cs.artifact_id = f.current_artifact_id
        ORDER BY f.depth, f.current_artifact_id;
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (source_resource_version_id,))
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


__all__ = ["LineageRepositoryMixin"]
