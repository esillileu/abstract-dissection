"""Pipeline run lifecycle and analysis profile resolution."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from .base import BaseRepository


class RunsRepositoryMixin(BaseRepository):
    """Operations for creating pipeline runs and resolving analysis profiles."""

    def resolve_analysis_run(
        self, *, profile: str | None = None, run_id: str | None = None
    ) -> str:
        """Resolve exactly one explicit or canonical analysis input run."""
        if profile and run_id:
            raise ValueError("profile and run_id are mutually exclusive")
        if run_id:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT run_id FROM pipeline_runs WHERE run_id = %s", (run_id,)
                )
                if cur.fetchone() is None:
                    raise ValueError(f"unknown corpus run: {run_id}")
            return run_id
        selected = profile or "confirmatory-50k"
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_id FROM analysis_profiles
                WHERE profile_key = %s AND is_active
                """,
                (selected,),
            )
            row = cur.fetchone()
        if row is None:
            raise ValueError(f"no active analysis profile: {selected}")
        return row[0]

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
        self._commit()
        return run_id

    def update_run_status(self, run_id: str, status: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE pipeline_runs
                SET status = %s, finished_at = %s
                WHERE run_id = %s;
                """,
                (
                    status,
                    datetime.now(UTC) if status in ("completed", "failed") else None,
                    run_id,
                ),
            )
        self._commit()
