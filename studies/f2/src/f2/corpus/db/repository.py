"""Corpus state repository implementing atomic transitions and queries for PostgreSQL."""

from __future__ import annotations

import json
from typing import Any

import psycopg


class CorpusStateRepository:
    """PostgreSQL-backed operational state repository for the F2 corpus pipeline."""

    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        self.conn = conn

    def _commit(self) -> None:
        """Commit unless an outer transaction owns the connection."""
        if not getattr(self.conn, "_num_transactions", 0):
            self.conn.commit()

    def create_acquisition_run(
        self,
        run_id: str,
        resource_version_id: str,
        acquisition_method: str,
        code_version: str,
        config_hash: str,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record the start of an external corpus acquisition run."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO acquisition_runs (
                    run_id, resource_version_id, acquisition_method, code_version,
                    config_hash, config, status, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, 'running', %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    status = 'running',
                    finished_at = NULL,
                    metadata = EXCLUDED.metadata;
                """,
                (
                    run_id,
                    resource_version_id,
                    acquisition_method,
                    code_version,
                    config_hash,
                    json.dumps(config or {}),
                    json.dumps(metadata or {}),
                ),
            )
        self._commit()
        return run_id

    def finish_acquisition_run(
        self,
        run_id: str,
        status: str,
        log_s3_uri: str | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark an acquisition run as completed or failed."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE acquisition_runs
                SET status = %s,
                    finished_at = NOW(),
                    log_s3_uri = COALESCE(%s, log_s3_uri),
                    error_message = %s,
                    metadata = CASE WHEN %s::jsonb IS NOT NULL THEN metadata || %s::jsonb ELSE metadata END
                WHERE run_id = %s;
                """,
                (
                    status,
                    log_s3_uri,
                    error_message,
                    json.dumps(metadata) if metadata is not None else None,
                    json.dumps(metadata) if metadata is not None else None,
                    run_id,
                ),
            )
        self._commit()

    def get_acquisition_run(self, run_id: str) -> dict[str, Any] | None:
        """Retrieve acquisition run details."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM acquisition_runs WHERE run_id = %s;", (run_id,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row, strict=False))

    def register_artifact(
        self,
        artifact_id: str,
        stage: str,
        s3_uri: str,
        sha256: str,
        byte_size: int,
        format: str,
        record_count: int | None = None,
        resource_version_id: str | None = None,
        acquisition_run_id: str | None = None,
        integrity_status: str = "pending",
        verification_report: dict[str, Any] | None = None,
    ) -> str:
        """Register an immutable artifact stored in SeaweedFS S3."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, stage, s3_uri, sha256, byte_size, format,
                    record_count, resource_version_id, acquisition_run_id,
                    integrity_status, verification_report
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (artifact_id) DO UPDATE SET
                    stage = EXCLUDED.stage,
                    s3_uri = EXCLUDED.s3_uri,
                    sha256 = EXCLUDED.sha256,
                    byte_size = EXCLUDED.byte_size,
                    format = EXCLUDED.format,
                    record_count = EXCLUDED.record_count,
                    integrity_status = EXCLUDED.integrity_status,
                    verification_report = EXCLUDED.verification_report;
                """,
                (
                    artifact_id,
                    stage,
                    s3_uri,
                    sha256,
                    byte_size,
                    format,
                    record_count,
                    resource_version_id,
                    acquisition_run_id,
                    integrity_status,
                    json.dumps(verification_report or {}),
                ),
            )
        self._commit()
        return artifact_id

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        """Retrieve artifact metadata by ID."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM artifacts WHERE artifact_id = %s;", (artifact_id,)
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row, strict=False))

    def update_artifact_verification(
        self,
        artifact_id: str,
        integrity_status: str,
        verification_report: dict[str, Any] | None = None,
    ) -> None:
        """Update integrity verification outcome for an artifact."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE artifacts
                SET integrity_status = %s,
                    verified_at = NOW(),
                    verification_report = %s
                WHERE artifact_id = %s;
                """,
                (integrity_status, json.dumps(verification_report or {}), artifact_id),
            )
        self._commit()

    def create_processing_run(
        self,
        run_id: str,
        recipe_name: str,
        recipe_version: str,
        code_version: str,
        config_hash: str,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record the start of a transformation / filtering / sharding processing run."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO processing_runs (
                    run_id, recipe_name, recipe_version, code_version,
                    config_hash, config, status, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, 'running', %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    status = 'running',
                    finished_at = NULL,
                    metadata = EXCLUDED.metadata;
                """,
                (
                    run_id,
                    recipe_name,
                    recipe_version,
                    code_version,
                    config_hash,
                    json.dumps(config or {}),
                    json.dumps(metadata or {}),
                ),
            )
        self._commit()
        return run_id

    def finish_processing_run(
        self,
        run_id: str,
        status: str,
        log_s3_uri: str | None = None,
        diagnostics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark a processing run as completed or failed."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE processing_runs
                SET status = %s,
                    finished_at = NOW(),
                    log_s3_uri = COALESCE(%s, log_s3_uri),
                    diagnostics = %s,
                    metadata = CASE WHEN %s::jsonb IS NOT NULL THEN metadata || %s::jsonb ELSE metadata END
                WHERE run_id = %s;
                """,
                (
                    status,
                    log_s3_uri,
                    json.dumps(diagnostics or {}),
                    json.dumps(metadata) if metadata is not None else None,
                    json.dumps(metadata) if metadata is not None else None,
                    run_id,
                ),
            )
        self._commit()

    def get_processing_run(self, run_id: str) -> dict[str, Any] | None:
        """Retrieve processing run details."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM processing_runs WHERE run_id = %s;", (run_id,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row, strict=False))

    def record_processing_io(
        self,
        run_id: str,
        inputs: list[tuple[str, str] | dict[str, str] | str],
        outputs: list[tuple[str, str] | dict[str, str] | str],
    ) -> None:
        """Record input and output artifacts with their roles for a processing run."""
        parsed_inputs: list[tuple[str, str]] = []
        for item in inputs:
            if isinstance(item, tuple):
                parsed_inputs.append((item[0], item[1]))
            elif isinstance(item, dict):
                parsed_inputs.append((item["artifact_id"], item.get("role", "primary")))
            else:
                parsed_inputs.append((str(item), "primary"))

        parsed_outputs: list[tuple[str, str]] = []
        for item in outputs:
            if isinstance(item, tuple):
                parsed_outputs.append((item[0], item[1]))
            elif isinstance(item, dict):
                parsed_outputs.append(
                    (item["artifact_id"], item.get("role", "primary"))
                )
            else:
                parsed_outputs.append((str(item), "primary"))

        with self.conn.cursor() as cur:
            for art_id, role in parsed_inputs:
                cur.execute(
                    """
                    INSERT INTO processing_run_inputs (run_id, artifact_id, input_role)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (run_id, artifact_id) DO UPDATE SET input_role = EXCLUDED.input_role;
                    """,
                    (run_id, art_id, role),
                )
            for art_id, role in parsed_outputs:
                cur.execute(
                    """
                    INSERT INTO processing_run_outputs (run_id, artifact_id, output_role)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (run_id, artifact_id) DO UPDATE SET output_role = EXCLUDED.output_role;
                    """,
                    (run_id, art_id, role),
                )
        self._commit()

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

    def register_corpus_shard(
        self,
        resource_version_id: str,
        shard_index: int,
        artifact_id: str,
        word_count: int = 0,
        doc_count: int = 0,
        byte_size: int = 0,
    ) -> None:
        """Bind an artifact to an ordered corpus shard of a published version."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO corpus_shards (
                    resource_version_id, shard_index, artifact_id, word_count, doc_count, byte_size
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (resource_version_id, shard_index) DO UPDATE SET
                    artifact_id = EXCLUDED.artifact_id,
                    word_count = EXCLUDED.word_count,
                    doc_count = EXCLUDED.doc_count,
                    byte_size = EXCLUDED.byte_size;
                """,
                (
                    resource_version_id,
                    shard_index,
                    artifact_id,
                    word_count,
                    doc_count,
                    byte_size,
                ),
            )
        self._commit()

    def upsert_corpus_version_stats(
        self,
        resource_version_id: str,
        total_words: int,
        total_tokens: int,
        total_documents: int,
        total_sentences: int,
        total_bytes: int,
        total_shards: int,
        source_composition: dict[str, Any] | None = None,
        year_distribution: dict[str, Any] | None = None,
    ) -> None:
        """Upsert aggregated NLP corpus statistics."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO corpus_version_stats (
                    resource_version_id, total_words, total_tokens, total_documents,
                    total_sentences, total_bytes, total_shards, source_composition,
                    year_distribution, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (resource_version_id) DO UPDATE SET
                    total_words = EXCLUDED.total_words,
                    total_tokens = EXCLUDED.total_tokens,
                    total_documents = EXCLUDED.total_documents,
                    total_sentences = EXCLUDED.total_sentences,
                    total_bytes = EXCLUDED.total_bytes,
                    total_shards = EXCLUDED.total_shards,
                    source_composition = EXCLUDED.source_composition,
                    year_distribution = EXCLUDED.year_distribution,
                    updated_at = NOW();
                """,
                (
                    resource_version_id,
                    total_words,
                    total_tokens,
                    total_documents,
                    total_sentences,
                    total_bytes,
                    total_shards,
                    json.dumps(source_composition or {}),
                    json.dumps(year_distribution or {}),
                ),
            )
        self._commit()

    def get_corpus_version_stats(
        self, resource_version_id: str
    ) -> dict[str, Any] | None:
        """Retrieve corpus version statistics."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM corpus_version_stats WHERE resource_version_id = %s;",
                (resource_version_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row, strict=False))

    def create_validation_profile(
        self,
        profile_id: str,
        profile_key: str,
        revision: int,
        name: str,
        spec_hash: str,
        specification: dict[str, Any],
        description: str | None = None,
        target_paper_id: str | None = None,
        notes: str | None = None,
    ) -> str:
        """Create an immutable validation profile specification."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO validation_profiles (
                    profile_id, profile_key, revision, name, description,
                    target_paper_id, spec_hash, specification, is_active, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, TRUE, %s);
                """,
                (
                    profile_id,
                    profile_key,
                    revision,
                    name,
                    description,
                    target_paper_id,
                    spec_hash,
                    json.dumps(specification),
                    notes,
                ),
            )
        self._commit()
        return profile_id

    def deprecate_validation_profile(
        self, profile_id: str, notes: str | None = None
    ) -> None:
        """Deprecate a validation profile revision (without modifying specification)."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE validation_profiles
                SET is_active = FALSE,
                    superseded_at = NOW(),
                    notes = COALESCE(%s, notes)
                WHERE profile_id = %s;
                """,
                (notes, profile_id),
            )
        self._commit()

    def get_validation_profile(self, profile_id: str) -> dict[str, Any] | None:
        """Retrieve validation profile by ID."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM validation_profiles WHERE profile_id = %s;",
                (profile_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row, strict=False))

    def record_validation_run(
        self,
        validation_run_id: str,
        profile_id: str,
        target_type: str,
        target_id: str,
        validator_code_version: str,
        validator_config_hash: str,
        overall_verdict: str,
        checks: list[dict[str, Any]],
        paper_compatibility: str = "not_applicable",
        validator_config: dict[str, Any] | None = None,
        evidence_report_uri: str | None = None,
        summary_metrics: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> str:
        """Record validation results for a processing run, artifact, or resource version."""
        target_proc_id = target_id if target_type == "processing_run" else None
        target_art_id = target_id if target_type == "artifact" else None
        target_rv_id = target_id if target_type == "resource_version" else None

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO validation_runs (
                    validation_run_id, profile_id, target_type,
                    target_processing_run_id, target_artifact_id, target_resource_version_id,
                    validator_code_version, validator_config_hash, validator_config,
                    overall_verdict, paper_compatibility, evidence_report_uri,
                    summary_metrics, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    validation_run_id,
                    profile_id,
                    target_type,
                    target_proc_id,
                    target_art_id,
                    target_rv_id,
                    validator_code_version,
                    validator_config_hash,
                    json.dumps(validator_config or {}),
                    overall_verdict,
                    paper_compatibility,
                    evidence_report_uri,
                    json.dumps(summary_metrics or {}),
                    notes,
                ),
            )
            for check in checks:
                cur.execute(
                    """
                    INSERT INTO validation_checks (
                        validation_run_id, check_name, category, status,
                        expected_condition, observed_value, details
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        validation_run_id,
                        check["check_name"],
                        check.get("category", "integrity"),
                        check["status"],
                        str(check.get("expected_condition", "")),
                        str(check.get("observed_value", "")),
                        json.dumps(check.get("details", {})),
                    ),
                )
        self._commit()
        return validation_run_id

    def get_validation_run(self, validation_run_id: str) -> dict[str, Any] | None:
        """Retrieve validation run along with its assertion checks."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM validation_runs WHERE validation_run_id = %s;",
                (validation_run_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [desc[0] for desc in cur.description]
            res = dict(zip(cols, row, strict=False))

            cur.execute(
                "SELECT * FROM validation_checks WHERE validation_run_id = %s ORDER BY check_id;",
                (validation_run_id,),
            )
            check_cols = [desc[0] for desc in cur.description]
            res["checks"] = [
                dict(zip(check_cols, c_row, strict=False)) for c_row in cur.fetchall()
            ]
            return res

    def get_validation_history(
        self, target_type: str, target_id: str
    ) -> list[dict[str, Any]]:
        """Retrieve validation history for a specific target ordered by execution time."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT vr.*, vp.profile_key, vp.revision AS profile_revision, vp.spec_hash
                FROM validation_runs vr
                JOIN validation_profiles vp ON vp.profile_id = vr.profile_id
                WHERE vr.target_type = %s AND vr.target_id = %s
                ORDER BY vr.started_at DESC;
                """,
                (target_type, target_id),
            )
            cols = [desc[0] for desc in cur.description]
            runs = [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]
            for run in runs:
                cur.execute(
                    "SELECT * FROM validation_checks WHERE validation_run_id = %s ORDER BY check_id;",
                    (run["validation_run_id"],),
                )
                c_cols = [desc[0] for desc in cur.description]
                run["checks"] = [
                    dict(zip(c_cols, c_row, strict=False)) for c_row in cur.fetchall()
                ]
            return runs


__all__ = ["CorpusStateRepository"]
