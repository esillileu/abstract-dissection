"""Pipeline run lifecycle."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from .base import BaseRepository


class RunsRepositoryMixin(BaseRepository):
    """Operations for creating pipeline runs."""

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
