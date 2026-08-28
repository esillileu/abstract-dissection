"""Repository layer providing transactional operations and view queries for F2 Catalog DB."""

from __future__ import annotations

import json
from typing import Any

import psycopg


class CatalogRepository:
    """PostgreSQL-backed repository for reproduction catalog entities and execution plans."""

    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        self.conn = conn

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

    # 4. Resources & Versions
    def upsert_resource(
        self,
        resource_id: str,
        kind: str,
        name: str,
        access_status: str,
        acquisition_status: str,
        readiness_status: str,
        description: str | None = None,
        canonical_version_id: str | None = None,
        notes: str | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO resources (
                    resource_id, kind, name, description, access_status,
                    acquisition_status, readiness_status, canonical_version_id, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (resource_id) DO UPDATE SET
                    kind = EXCLUDED.kind,
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    access_status = EXCLUDED.access_status,
                    acquisition_status = EXCLUDED.acquisition_status,
                    readiness_status = EXCLUDED.readiness_status,
                    canonical_version_id = EXCLUDED.canonical_version_id,
                    notes = EXCLUDED.notes;
                """,
                (
                    resource_id,
                    kind,
                    name,
                    description,
                    access_status,
                    acquisition_status,
                    readiness_status,
                    canonical_version_id,
                    notes,
                ),
            )

    def set_resource_canonical_version(
        self,
        resource_id: str,
        resource_version_id: str | None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE resources
                SET canonical_version_id = %s
                WHERE resource_id = %s;
                """,
                (resource_version_id, resource_id),
            )

    def upsert_resource_version(
        self,
        resource_version_id: str,
        resource_id: str,
        version_label: str | None = None,
        uri: str | None = None,
        local_path: str | None = None,
        checksum_algo: str | None = None,
        checksum: str | None = None,
        size_bytes: int | None = None,
        metadata: dict[str, Any] | None = None,
        is_verified: bool = False,
        notes: str | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO resource_versions (
                    resource_version_id, resource_id, version_label, uri, local_path,
                    checksum_algo, checksum, size_bytes, metadata, is_verified, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (resource_version_id) DO UPDATE SET
                    resource_id = EXCLUDED.resource_id,
                    version_label = EXCLUDED.version_label,
                    uri = EXCLUDED.uri,
                    local_path = EXCLUDED.local_path,
                    checksum_algo = EXCLUDED.checksum_algo,
                    checksum = EXCLUDED.checksum,
                    size_bytes = EXCLUDED.size_bytes,
                    metadata = EXCLUDED.metadata,
                    is_verified = EXCLUDED.is_verified,
                    notes = EXCLUDED.notes;
                """,
                (
                    resource_version_id,
                    resource_id,
                    version_label,
                    uri,
                    local_path,
                    checksum_algo,
                    checksum,
                    size_bytes,
                    json.dumps(metadata or {}),
                    is_verified,
                    notes,
                ),
            )

    def upsert_resource_source(
        self,
        resource_id: str,
        source_type: str,
        url: str | None = None,
        citation: str | None = None,
        license: str | None = None,
        is_preferred: bool = False,
        notes: str | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            if is_preferred:
                # Clear previous preferred source for this resource if set
                cur.execute(
                    """
                    UPDATE resource_sources
                    SET is_preferred = FALSE
                    WHERE resource_id = %s AND is_preferred = TRUE;
                    """,
                    (resource_id,),
                )
            cur.execute(
                """
                INSERT INTO resource_sources (
                    resource_id, source_type, url, citation, license, is_preferred, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    resource_id,
                    source_type,
                    url,
                    citation,
                    license,
                    is_preferred,
                    notes,
                ),
            )

    def upsert_preparation_spec(
        self,
        preparation_id: str,
        input_resource_id: str,
        preparation_type: str,
        specification: dict[str, Any],
        output_resource_id: str | None = None,
        code_resource_id: str | None = None,
        status: str = "planned",
        notes: str | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO preparation_specs (
                    preparation_id, input_resource_id, output_resource_id,
                    preparation_type, specification, code_resource_id, status, notes
                ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                ON CONFLICT (preparation_id) DO UPDATE SET
                    input_resource_id = EXCLUDED.input_resource_id,
                    output_resource_id = EXCLUDED.output_resource_id,
                    preparation_type = EXCLUDED.preparation_type,
                    specification = EXCLUDED.specification,
                    code_resource_id = EXCLUDED.code_resource_id,
                    status = EXCLUDED.status,
                    notes = EXCLUDED.notes;
                """,
                (
                    preparation_id,
                    input_resource_id,
                    output_resource_id,
                    preparation_type,
                    json.dumps(specification),
                    code_resource_id,
                    status,
                    notes,
                ),
            )

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

    # 6. Execution Plans & Experiments
    def upsert_execution_plan(
        self,
        execution_plan_id: str,
        plan_key: str,
        revision: int,
        status: str = "draft",
        source_ref: str | None = None,
        source_hash: str | None = None,
        is_canonical: bool = False,
        notes: str | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            if is_canonical:
                cur.execute(
                    """
                    UPDATE execution_plans
                    SET is_canonical = FALSE, superseded_at = NOW()
                    WHERE plan_key = %s AND is_canonical = TRUE AND execution_plan_id <> %s;
                    """,
                    (plan_key, execution_plan_id),
                )
            cur.execute(
                """
                INSERT INTO execution_plans (
                    execution_plan_id, plan_key, revision, status,
                    source_ref, source_hash, is_canonical, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (execution_plan_id) DO UPDATE SET
                    plan_key = EXCLUDED.plan_key,
                    revision = EXCLUDED.revision,
                    status = EXCLUDED.status,
                    source_ref = EXCLUDED.source_ref,
                    source_hash = EXCLUDED.source_hash,
                    is_canonical = EXCLUDED.is_canonical,
                    notes = EXCLUDED.notes;
                """,
                (
                    execution_plan_id,
                    plan_key,
                    revision,
                    status,
                    source_ref,
                    source_hash,
                    is_canonical,
                    notes,
                ),
            )

    def set_canonical_execution_plan(
        self,
        execution_plan_id: str,
        plan_key: str,
    ) -> None:
        """Atomically demote previous canonical revision and promote the specified plan."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE execution_plans
                SET is_canonical = FALSE, superseded_at = NOW()
                WHERE plan_key = %s AND is_canonical = TRUE;
                """,
                (plan_key,),
            )
            cur.execute(
                """
                UPDATE execution_plans
                SET is_canonical = TRUE, superseded_at = NULL
                WHERE execution_plan_id = %s;
                """,
                (execution_plan_id,),
            )

    def upsert_plan_experiment(
        self,
        plan_experiment_id: str,
        execution_plan_id: str,
        experiment_spec_id: str,
        enabled: bool = True,
        parameters: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO execution_plan_experiments (
                    plan_experiment_id, execution_plan_id, experiment_spec_id,
                    enabled, parameters, notes
                ) VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (plan_experiment_id) DO UPDATE SET
                    execution_plan_id = EXCLUDED.execution_plan_id,
                    experiment_spec_id = EXCLUDED.experiment_spec_id,
                    enabled = EXCLUDED.enabled,
                    parameters = EXCLUDED.parameters,
                    notes = EXCLUDED.notes;
                """,
                (
                    plan_experiment_id,
                    execution_plan_id,
                    experiment_spec_id,
                    enabled,
                    json.dumps(parameters or {}),
                    notes,
                ),
            )

    def bind_plan_requirement(
        self,
        plan_experiment_id: str,
        requirement_id: str,
        resource_version_id: str,
        binding_type: str,
        justification: str | None = None,
        notes: str | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO execution_plan_bindings (
                    plan_experiment_id, requirement_id, resource_version_id,
                    binding_type, justification, notes
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (plan_experiment_id, requirement_id) DO UPDATE SET
                    resource_version_id = EXCLUDED.resource_version_id,
                    binding_type = EXCLUDED.binding_type,
                    justification = EXCLUDED.justification,
                    notes = EXCLUDED.notes;
                """,
                (
                    plan_experiment_id,
                    requirement_id,
                    resource_version_id,
                    binding_type,
                    justification,
                    notes,
                ),
            )

    # 7. Planned Run Slots
    def upsert_planned_run_slot(
        self,
        planned_run_slot_id: str,
        plan_experiment_id: str,
        slot_key: str,
        atomic_run_id: str | None = None,
        variant_key: str | None = None,
        seed: int | None = None,
        parameters: dict[str, Any] | None = None,
        expected: bool = True,
        reference_mlflow_run_id: str | None = None,
        notes: str | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO planned_run_slots (
                    planned_run_slot_id, plan_experiment_id, slot_key,
                    atomic_run_id, variant_key, seed, parameters,
                    expected, reference_mlflow_run_id, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                ON CONFLICT (plan_experiment_id, slot_key) DO UPDATE SET
                    atomic_run_id = EXCLUDED.atomic_run_id,
                    variant_key = EXCLUDED.variant_key,
                    seed = EXCLUDED.seed,
                    parameters = EXCLUDED.parameters,
                    expected = EXCLUDED.expected,
                    reference_mlflow_run_id = COALESCE(EXCLUDED.reference_mlflow_run_id, planned_run_slots.reference_mlflow_run_id),
                    notes = EXCLUDED.notes;
                """,
                (
                    planned_run_slot_id,
                    plan_experiment_id,
                    slot_key,
                    atomic_run_id,
                    variant_key,
                    seed,
                    json.dumps(parameters or {}),
                    expected,
                    reference_mlflow_run_id,
                    notes,
                ),
            )

    def link_mlflow_run(
        self,
        planned_run_slot_id: str,
        reference_mlflow_run_id: str | None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE planned_run_slots
                SET reference_mlflow_run_id = %s
                WHERE planned_run_slot_id = %s;
                """,
                (reference_mlflow_run_id, planned_run_slot_id),
            )

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
