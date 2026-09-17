"""Reported results and analytical view queries for F2 Catalog DB."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseCatalogRepository


class ViewsRepositoryMixin(BaseCatalogRepository):
    """Operations for recording reported results and querying catalog views."""

    # 8. Reported Results
    def record_reported_result(
        self,
        target_id: str,
        metric: str,
        value: float | None = None,
        value_text: str | None = None,
        unit: str | None = None,
        aggregation: str | None = None,
        metadata: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reported_results (
                    target_id, metric, value, value_text, unit,
                    aggregation, metadata, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                RETURNING reported_result_id;
                """,
                (
                    target_id,
                    metric,
                    value,
                    value_text,
                    unit,
                    aggregation,
                    json.dumps(metadata or {}),
                    notes,
                ),
            )
            return cur.fetchone()[0]

    # 9. View Queries
    def get_canonical_run_matrix(
        self, plan_key: str | None = None
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM v_canonical_run_matrix"
        params: list[Any] = []
        if plan_key:
            query += " WHERE plan_key = %s"
            params.append(plan_key)
        query += " ORDER BY experiment_spec_id, slot_key;"

        with self.conn.cursor() as cur:
            cur.execute(query, params)
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]

    def get_canonical_plan_progress(
        self, plan_key: str | None = None
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM v_canonical_plan_progress"
        params: list[Any] = []
        if plan_key:
            query += " WHERE plan_key = %s"
            params.append(plan_key)
        query += " ORDER BY experiment_spec_id;"

        with self.conn.cursor() as cur:
            cur.execute(query, params)
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]

    def get_resource_inventory(self) -> list[dict[str, Any]]:
        query = "SELECT * FROM v_resource_inventory ORDER BY kind, name;"
        with self.conn.cursor() as cur:
            cur.execute(query)
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]
