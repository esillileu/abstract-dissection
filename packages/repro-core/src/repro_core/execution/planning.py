"""Catalog traversal and expansion into atomic seeded runs."""

from __future__ import annotations

import yaml

from .definition import ExecutionDefinition, RunOptions, RunOrder, RunPlan, RunSelection
from .parsing import (
    deep_merge,
    parse_atomic_run_ids,
    parse_experiment_ids,
    parse_seed_values,
)


class Planner:
    def __init__(self, domain: ExecutionDefinition) -> None:
        self.domain = domain

    def build(self, selection: RunSelection, options: RunOptions) -> list[RunPlan]:
        if selection.experiment_ids and selection.all_experiments:
            raise ValueError("choose at most one of --all or --experiment/-e")
        if selection.atomic_run_ids and selection.excluded_atomic_run_ids:
            raise ValueError(
                "choose at most one of --atomic-run/-a or --exclude-atomic-run/-x"
            )
        selected = set(parse_experiment_ids(list(selection.experiment_ids)))
        included = set(parse_atomic_run_ids(list(selection.atomic_run_ids)))
        excluded = set(parse_atomic_run_ids(list(selection.excluded_atomic_run_ids)))
        requested_atomic_runs = included or excluded
        matched_atomic_runs: set[str] = set()
        seeds = self._seed_values(selection.seed_set or self.domain.default_seed_set)
        requested_seeds = parse_seed_values(selection.seed_values, available=seeds)
        ordered_seeds = requested_seeds if requested_seeds is not None else seeds
        seed_order = {seed: position for position, seed in enumerate(ordered_seeds)}
        plans: list[RunPlan] = []
        matched_experiments: set[str] = set()
        for path in sorted(self.domain.config_root.glob("e[0-9][0-9]_*.yaml")):
            experiment_id = path.name[:3]
            if not selection.all_experiments and selected:
                if experiment_id not in selected:
                    continue
            elif not selection.all_experiments and not selected:
                continue
            matched_experiments.add(experiment_id)
            source = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(source, dict):
                raise ValueError(f"invalid YAML object: {path}")
            if (
                selection.all_experiments
                and str(source.get("kind", "")) in self.domain.all_excluded_kinds
            ):
                continue
            resolved = deep_merge(source, options.overrides)
            variants = resolved.get("variants")
            if not isinstance(variants, dict) or not variants:
                raise ValueError(f"experiment YAML needs variants: {path}")
            available_atomic_runs = {str(item) for item in variants}
            matched_atomic_runs.update(requested_atomic_runs & available_atomic_runs)
            selected_variants = [
                str(item)
                for item in variants
                if (not included or str(item) in included) and str(item) not in excluded
            ]
            if not selected_variants:
                continue
            for atomic_run_id in selected_variants:
                variant = variants[atomic_run_id]
                if not isinstance(variant, dict):
                    raise ValueError(
                        f"experiment variant must be a mapping: {path}/{atomic_run_id}"
                    )
                atomic = deep_merge(resolved, variant)
                execution = atomic.get("execution", {})
                if not isinstance(execution, dict):
                    raise ValueError(f"execution must be a mapping: {path}")
                mode = str(execution.get("mode", "seeded"))
                if mode not in {"seeded", "single"}:
                    raise ValueError(f"unsupported execution.mode in {path}: {mode}")
                if mode == "single":
                    if requested_seeds is not None:
                        raise ValueError(
                            f"{experiment_id} is a single-run experiment and "
                            "does not accept --seed"
                        )
                    run_seeds: list[int | None] = [None]
                else:
                    policy = atomic.get("seed_policy", {})
                    if not isinstance(policy, dict):
                        raise ValueError(f"seed_policy must be a mapping: {path}")
                    count = int(policy.get("seed_count", len(seeds)))
                    available_seeds = seeds[:count]
                    run_seeds = (
                        requested_seeds
                        if requested_seeds is not None
                        else available_seeds
                    )
                    invalid = [
                        seed for seed in run_seeds if seed not in available_seeds
                    ]
                    if invalid:
                        raise ValueError(
                            f"{experiment_id}/{atomic_run_id} declares seed "
                            f"values {available_seeds}; invalid values: {invalid}"
                        )
                for seed in run_seeds:
                    plans.append(
                        RunPlan(
                            self.domain.name,
                            experiment_id,
                            path,
                            atomic_run_id,
                            seed,
                            options.device
                            or str(execution.get("default_device", "cpu")),
                        )
                    )
        unknown_experiments = selected - matched_experiments
        if unknown_experiments:
            raise ValueError(
                "unknown experiment in catalog: "
                + ", ".join(sorted(unknown_experiments))
            )
        unknown_atomic_runs = requested_atomic_runs - matched_atomic_runs
        if unknown_atomic_runs:
            raise ValueError(
                "unknown atomic run ID in selected experiments: "
                + ", ".join(sorted(unknown_atomic_runs))
            )
        if not plans:
            if requested_atomic_runs:
                raise ValueError("atomic run selection matched no plans")
            raise ValueError("no experiment YAML matched")
        if options.order is RunOrder.SEED_FIRST:
            plans.sort(
                key=lambda plan: (
                    seed_order.get(plan.seed, len(seed_order))
                    if plan.seed is not None
                    else len(seed_order)
                )
            )
        return plans

    def _seed_values(self, seed_set: str) -> list[int]:
        path = self.domain.config_root / "seeds.yaml"
        registry = yaml.safe_load(path.read_text(encoding="utf-8"))
        try:
            return [int(value) for value in registry["seed_sets"][seed_set]["values"]]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"unknown seed set for {self.domain.name}: {seed_set}"
            ) from exc
