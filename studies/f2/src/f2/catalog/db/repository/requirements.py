"""Experiment requirements and candidate bindings for F2 Catalog DB."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseCatalogRepository


class RequirementsRepositoryMixin(BaseCatalogRepository):
    """Transactional operations for experiment requirements and candidates."""

    # 5. Requirements & Candidates
    def upsert_requirement(
        self,
        requirement_id: str,
        experiment_spec_id: str,
        role: str,
        required_resource_id: str | None = None,
        requirement_spec: dict[str, Any] | None = None,
        required: bool = True,
        notes: str | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO experiment_requirements (
                    requirement_id, experiment_spec_id, role, required_resource_id,
                    requirement_spec, required, notes
                ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (requirement_id) DO UPDATE SET
                    experiment_spec_id = EXCLUDED.experiment_spec_id,
                    role = EXCLUDED.role,
                    required_resource_id = EXCLUDED.required_resource_id,
                    requirement_spec = EXCLUDED.requirement_spec,
                    required = EXCLUDED.required,
                    notes = EXCLUDED.notes;
                """,
                (
                    requirement_id,
                    experiment_spec_id,
                    role,
                    required_resource_id,
                    json.dumps(requirement_spec) if requirement_spec else None,
                    required,
                    notes,
                ),
            )

    def upsert_requirement_candidate(
        self,
        requirement_id: str,
        resource_id: str,
        candidate_type: str,
        status: str,
        justification: str | None = None,
        notes: str | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO requirement_candidates (
                    requirement_id, resource_id, candidate_type, status, justification, notes
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (requirement_id, resource_id) DO UPDATE SET
                    candidate_type = EXCLUDED.candidate_type,
                    status = EXCLUDED.status,
                    justification = EXCLUDED.justification,
                    notes = EXCLUDED.notes;
                """,
                (
                    requirement_id,
                    resource_id,
                    candidate_type,
                    status,
                    justification,
                    notes,
                ),
            )
