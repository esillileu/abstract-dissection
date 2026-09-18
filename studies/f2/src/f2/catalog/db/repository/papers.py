"""Papers, reproduction targets, and experiment specification operations for F2 Catalog DB."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseCatalogRepository


class PapersRepositoryMixin(BaseCatalogRepository):
    """Transactional operations for papers, reproduction targets, and experiment specs."""

    # 1. Papers
    def upsert_paper(
        self,
        paper_id: str,
        title: str,
        citation_key: str | None = None,
        version: str | None = None,
        source_url: str | None = None,
        notes: str | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO papers (paper_id, title, citation_key, version, source_url, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (paper_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    citation_key = EXCLUDED.citation_key,
                    version = EXCLUDED.version,
                    source_url = EXCLUDED.source_url,
                    notes = EXCLUDED.notes;
                """,
                (paper_id, title, citation_key, version, source_url, notes),
            )

    # 2. Reproduction Targets
    def upsert_target(
        self,
        target_id: str,
        paper_id: str,
        location_type: str,
        location_label: str,
        target_type: str,
        description: str,
        ordinal: int | None = None,
        source_locator: str | None = None,
        notes: str | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reproduction_targets (
                    target_id, paper_id, location_type, location_label, ordinal,
                    target_type, description, source_locator, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (target_id) DO UPDATE SET
                    paper_id = EXCLUDED.paper_id,
                    location_type = EXCLUDED.location_type,
                    location_label = EXCLUDED.location_label,
                    ordinal = EXCLUDED.ordinal,
                    target_type = EXCLUDED.target_type,
                    description = EXCLUDED.description,
                    source_locator = EXCLUDED.source_locator,
                    notes = EXCLUDED.notes;
                """,
                (
                    target_id,
                    paper_id,
                    location_type,
                    location_label,
                    ordinal,
                    target_type,
                    description,
                    source_locator,
                    notes,
                ),
            )

    # 3. Experiment Specs & Links
    def upsert_experiment_spec(
        self,
        experiment_spec_id: str,
        paper_id: str,
        name: str,
        run_type: str,
        provenance_status: str,
        specification: dict[str, Any] | None = None,
        source_locator: str | None = None,
        notes: str | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO experiment_specs (
                    experiment_spec_id, paper_id, name, run_type, provenance_status,
                    specification, source_locator, notes
                ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (experiment_spec_id) DO UPDATE SET
                    paper_id = EXCLUDED.paper_id,
                    name = EXCLUDED.name,
                    run_type = EXCLUDED.run_type,
                    provenance_status = EXCLUDED.provenance_status,
                    specification = EXCLUDED.specification,
                    source_locator = EXCLUDED.source_locator,
                    notes = EXCLUDED.notes;
                """,
                (
                    experiment_spec_id,
                    paper_id,
                    name,
                    run_type,
                    provenance_status,
                    json.dumps(specification or {}),
                    source_locator,
                    notes,
                ),
            )

    def link_target_experiment(
        self,
        target_id: str,
        experiment_spec_id: str,
        role: str = "primary",
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO target_experiments (target_id, experiment_spec_id, role)
                VALUES (%s, %s, %s)
                ON CONFLICT (target_id, experiment_spec_id) DO UPDATE SET
                    role = EXCLUDED.role;
                """,
                (target_id, experiment_spec_id, role),
            )
