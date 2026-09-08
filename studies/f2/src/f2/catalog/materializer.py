"""Adapter materializing repro-core execution planner definitions into Catalog DB planned run slots."""

from __future__ import annotations

import hashlib

from repro_core.execution.definition import (
    ExecutionDefinition,
    RunOptions,
    RunPlan,
    RunSelection,
)
from repro_core.execution.planning import Planner

from .db.repository import CatalogRepository


class CatalogPlanMaterializer:
    """Materializes atomic run/seed combinations from repro-core Planner into Catalog DB."""

    def __init__(self, repo: CatalogRepository) -> None:
        self.repo = repo

    def materialize_from_plans(
        self,
        plan_key: str,
        revision: int,
        plans: list[RunPlan],
        is_canonical: bool = True,
        status: str = "runnable",
        source_ref: str | None = None,
        notes: str | None = None,
    ) -> str:
        """Materialize a pre-expanded list of RunPlans into an execution plan and planned run slots."""
        execution_plan_id = f"{plan_key}_r{revision}"

        # Compute deterministic digest of planned runs
        hasher = hashlib.sha256()
        for p in sorted(
            plans, key=lambda x: (x.experiment_id, x.atomic_run_id, x.seed or 0)
        ):
            hasher.update(
                f"{p.experiment_id}:{p.atomic_run_id}:{p.seed}:{p.device}".encode()
            )
        source_hash = hasher.hexdigest()[:16]

        self.repo.upsert_execution_plan(
            execution_plan_id=execution_plan_id,
            plan_key=plan_key,
            revision=revision,
            status=status,
            source_ref=source_ref,
            source_hash=source_hash,
            is_canonical=is_canonical,
            notes=notes,
        )

        # Group by experiment
        by_exp: dict[str, list[RunPlan]] = {}
        for p in plans:
            by_exp.setdefault(p.experiment_id, []).append(p)

        for exp_id, exp_plans in by_exp.items():
            plan_exp_id = f"{execution_plan_id}_{exp_id}"
            spec_id = f"{plan_key}_{exp_id}"

            self.repo.upsert_plan_experiment(
                plan_experiment_id=plan_exp_id,
                execution_plan_id=execution_plan_id,
                experiment_spec_id=spec_id,
                enabled=True,
                parameters={"experiment_id": exp_id},
            )

            for p in exp_plans:
                seed_tag = f"s{p.seed}" if p.seed is not None else "single"
                slot_key = f"{p.atomic_run_id}__{seed_tag}"
                slot_id = f"{plan_exp_id}__{slot_key}"

                self.repo.upsert_planned_run_slot(
                    planned_run_slot_id=slot_id,
                    plan_experiment_id=plan_exp_id,
                    slot_key=slot_key,
                    atomic_run_id=p.atomic_run_id,
                    variant_key=p.atomic_run_id,
                    seed=p.seed,
                    parameters={"device": p.device, "path": str(p.path)},
                    expected=True,
                )

        return execution_plan_id

    def materialize_from_definition(
        self,
        domain: ExecutionDefinition,
        plan_key: str | None = None,
        revision: int = 1,
        is_canonical: bool = True,
        selection: RunSelection | None = None,
        options: RunOptions | None = None,
        notes: str | None = None,
    ) -> str:
        """Expand an ExecutionDefinition catalog into RunPlans using repro-core Planner, then persist."""
        key = plan_key or domain.name
        planner = Planner(domain)
        sel = selection or RunSelection(all_experiments=True)
        opts = options or RunOptions()
        plans = planner.build(sel, opts)

        return self.materialize_from_plans(
            plan_key=key,
            revision=revision,
            plans=plans,
            is_canonical=is_canonical,
            source_ref=str(domain.config_root),
            notes=notes,
        )
