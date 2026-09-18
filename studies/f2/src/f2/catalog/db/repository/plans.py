"""Execution plans, plan experiments, requirement bindings, and run slot operations for F2 Catalog DB."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseCatalogRepository


class PlansRepositoryMixin(BaseCatalogRepository):
    """Transactional operations for execution plans, experiments, bindings, and run slots."""

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
