from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MetricValue = int | float


@dataclass(frozen=True)
class RunIdentity:
    """Schema v1 identity for one MLflow seed_trial run."""

    schema_version: int
    project_name: str
    experiment_ids: tuple[str, ...]
    atomic_run_id: str
    execution_group_id: str
    recipe_id: str
    structure_signature: str
    condition_key: str
    run_key: str
    master_seed: int


class MLflowRunLogger:
    """Thin MLflow client wrapper with lazy import."""

    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment_name: str,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._mlflow = None
        self._active_run = None

    def start_child(
        self,
        *,
        run_name: str,
        tags: dict[str, str],
        params: dict[str, object],
    ) -> None:
        mlflow = self._import_mlflow()
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._active_run = mlflow.start_run(run_name=run_name, tags=tags)
        self.log_params(params)

    def log_tags(self, tags: dict[str, str]) -> None:
        self._import_mlflow().set_tags(tags)

    def log_params(self, params: dict[str, object]) -> None:
        cleaned = {
            key: _stringify_param(value)
            for key, value in params.items()
            if value is not None
        }
        if cleaned:
            self._import_mlflow().log_params(cleaned)

    def log_update_metrics(
        self,
        step: int,
        metrics: dict[str, MetricValue],
    ) -> None:
        self._log_metrics(metrics, step=step)

    def log_epoch_metrics(
        self,
        epoch: int,
        metrics: dict[str, MetricValue],
    ) -> None:
        self._log_metrics(metrics, step=epoch)

    def log_final_metrics(self, metrics: dict[str, MetricValue]) -> None:
        self._log_metrics(metrics, step=0)

    def log_artifact_tree(self, root: Path) -> None:
        self._import_mlflow().log_artifacts(str(root))

    def finalize_success(self) -> None:
        mlflow = self._import_mlflow()
        mlflow.set_tag("trial.status", "finished")
        mlflow.end_run(status="FINISHED")

    def finalize_failure(self, exc: BaseException) -> None:
        mlflow = self._import_mlflow()
        mlflow.set_tags(
            {
                "trial.status": "failed",
                "failure.type": "exception",
                "failure.message": str(exc),
            }
        )
        mlflow.end_run(status="FAILED")

    def _log_metrics(
        self,
        metrics: dict[str, MetricValue],
        *,
        step: int,
    ) -> None:
        cleaned = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        if cleaned:
            self._import_mlflow().log_metrics(cleaned, step=step)

    def _import_mlflow(self):
        if self._mlflow is not None:
            return self._mlflow

        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                "MLflow logging requires the tracking extra. "
                "Install with `uv sync --extra tracking`."
            ) from exc

        self._mlflow = mlflow
        return mlflow


def canonical_json(value: Any) -> str:
    """Serialize JSON in the schema's canonical digest format."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def make_condition_key(condition_config: dict[str, Any]) -> str:
    """Hash a seed-free condition config."""

    return _sha256_text(canonical_json(condition_config))


def make_run_key(
    condition_config: dict[str, Any],
    seed_config: dict[str, Any],
) -> str:
    """Hash a condition config and one seed config."""

    return _sha256_text(canonical_json(condition_config) + canonical_json(seed_config))


def flatten_dict(
    value: dict[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten nested dictionaries with slash-separated keys."""

    items: dict[str, Any] = {}
    for key, child in value.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(child, dict):
            items.update(flatten_dict(child, prefix=name))
        else:
            items[name] = child
    return items


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stringify_param(value: object) -> str:
    if isinstance(value, (dict, list, tuple)):
        return canonical_json(value)
    return str(value)
