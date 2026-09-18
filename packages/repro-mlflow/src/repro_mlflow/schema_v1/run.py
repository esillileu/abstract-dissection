"""Schema-v1 run orchestration and execution setup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from repro_core.config import normalize_config

from ..runtime import (
    ExperimentRun,
    RuntimeOptions,
    current_git_info,
    flatten_dict,
)
from .artifacts import write_run_artifacts
from .identity import (
    _section,
    _storage_domain,
    build_condition_config,
    build_identity,
    build_tags,
    seed_config,
)


class SchemaV1Run:
    """Owns the full legacy-compatible record for one YAML experiment."""

    def __init__(self, config: dict[str, object], *, tracking_uri: str) -> None:
        self.config = normalize_config(config)
        self.tracking_uri = tracking_uri.rstrip("/")
        if not self.tracking_uri:
            raise ValueError("SchemaV1Run requires a tracking URI")
        self.seed = int(self.config["seed"])
        self.seeds = seed_config(self.seed)
        policy = _section(self.config, "policy")
        if policy.get("dataset_split_seed") is not None:
            self.seeds["dataset_split"] = int(policy["dataset_split_seed"])
        self.git_info = current_git_info(
            str(_section(self.config, "training")["entrypoint"])
        )
        self.condition = build_condition_config(self.config, self.git_info)
        self.identity = build_identity(self.config, self.condition, self.seeds)
        tracking = _section(self.config, "tracking")
        experiment = tracking.get("experiment")
        if not experiment:
            raise ValueError("tracking.experiment is required")
        if tracking.get("enabled") is False:
            raise ValueError("tracking cannot be disabled")
        self.storage_domain = _storage_domain(experiment)
        from repro_core.context.paths import WorkspacePaths

        declared_tags = tracking.get("tags", {})
        tags = declared_tags if isinstance(declared_tags, dict) else {}
        domain = str(tags.get("domain.name", self.storage_domain))
        suite = str(tags.get("suite.name", self.storage_domain))
        study = str(self.identity.experiment_ids[0])
        variant = str(tags.get("implementation.variant", "implemented"))
        staging = WorkspacePaths.from_environment(Path.cwd()).run_staging(
            domain=domain,
            suite=suite,
            study=study,
            variant=variant,
            run_key=self.identity.run_key,
        )
        self.artifact_root = staging / "record"
        self.local_checkpoint_root = staging / "checkpoints"

    def runtime(self, *, model: Any | None = None) -> ExperimentRun:
        tracking = _section(self.config, "tracking")
        return ExperimentRun(
            options=RuntimeOptions(
                tracking_uri=self.tracking_uri,
                experiment_name=str(tracking["experiment"]),
                upload_checkpoint=bool(tracking.get("upload_checkpoint", True)),
                upload_eval_checkpoints=bool(
                    tracking.get("upload_eval_checkpoints", True)
                ),
            ),
            run_name=f"{self.identity.atomic_run_id}-s{self.identity.master_seed:02d}",
            tags=build_tags(self.identity, self.config, self.git_info, model),
            params=flatten_dict(
                {
                    **self.condition,
                    "seed": self.seeds,
                    "policy": _section(self.config, "policy"),
                    "regularization": _section(self.config, "regularization"),
                }
            ),
        )

    def write_artifacts(
        self,
        *,
        model: Any | None,
        final_metrics: dict[str, float],
        metric_rows: list[tuple[int, str, float]],
        profiling_metrics: dict[str, int | float],
        reproducibility: dict[str, object] | None = None,
        evaluation_checkpoints: list[Path] | None = None,
    ) -> None:
        write_run_artifacts(
            config=self.config,
            condition=self.condition,
            seeds=self.seeds,
            identity_condition_key=self.identity.condition_key,
            identity_run_key=self.identity.run_key,
            git_info=self.git_info,
            artifact_root=self.artifact_root,
            local_checkpoint_root=self.local_checkpoint_root,
            model=model,
            final_metrics=final_metrics,
            metric_rows=metric_rows,
            profiling_metrics=profiling_metrics,
            reproducibility=reproducibility,
            evaluation_checkpoints=evaluation_checkpoints,
        )
