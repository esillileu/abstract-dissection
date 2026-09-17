"""Processing runs and DAG I/O recording operations."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseCorpusRepository


class ProcessingRepositoryMixin(BaseCorpusRepository):
    """Repository operations for transformation / filtering / sharding processing runs."""

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


__all__ = ["ProcessingRepositoryMixin"]
