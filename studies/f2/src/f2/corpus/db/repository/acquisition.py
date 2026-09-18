"""Acquisition run tracking and state operations."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseCorpusRepository


class AcquisitionRepositoryMixin(BaseCorpusRepository):
    """Repository operations for external corpus source acquisition runs."""

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


__all__ = ["AcquisitionRepositoryMixin"]
