"""Publication service for publishing F2 results to MLflow.

Implements ref.md section 4.1 & 4.4:
- F2 -> MLflow atomic publication
- Querying finished runs strictly by ConditionIdentity / tags
- Native MLflow FINISHED status as sole completion criterion
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from mlflow import MlflowClient
from mlflow.entities import Metric, Run

from f2.execution.identity import ConditionIdentity


class PublicationService:
    """Manages publication of verified local run artifacts to MLflow."""

    def __init__(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri=tracking_uri)

    def find_matching_finished_run(
        self,
        experiment_name: str,
        condition: ConditionIdentity,
        *,
        atomic_run_id: str | None = None,
    ) -> Run | None:
        """Find an existing FINISHED run matching the exact condition identity."""
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None

        runs = self.client.search_runs(
            [experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            max_results=10000,
        )
        for run in runs:
            tags = run.data.tags
            digest = tags.get("f2.config_digest") or tags.get("config_digest")
            run_seed = tags.get("f2.seed") or tags.get("seed")

            if digest != condition.config_digest or run_seed != str(condition.seed):
                continue

            if condition.planned_run_slot_id:
                slot = tags.get("f2.planned_run_slot_id") or tags.get(
                    "planned_run_slot_id"
                )
                if slot and slot != condition.planned_run_slot_id:
                    continue

            if atomic_run_id:
                run_variant = tags.get("atomic_run_id") or tags.get("f2.atomic_run_id")
                if run_variant and run_variant != atomic_run_id:
                    continue

            return run
        return None

    def publish_run(
        self,
        experiment_name: str,
        condition: ConditionIdentity,
        run_root: Path,
        config: dict[str, Any],
        *,
        model_family: str = "word2vec",
    ) -> str:
        """Publish local execution outputs to MLflow and set FINISHED."""
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = self.client.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        # Merge standard condition tags with execution tags
        tags = condition.tags()
        tags["atomic_run_id"] = str(config.get("atomic_run_id", ""))
        tags["f2.atomic_run_id"] = tags["atomic_run_id"]
        tags["f2.model_family"] = model_family

        mlflow_run = self.client.create_run(
            experiment_id=experiment_id,
            tags=tags,
            run_name=condition.planned_run_slot_id,
        )
        run_id = mlflow_run.info.run_id

        try:
            self._log_parameters(run_id, config)
            self._log_metrics(run_id, run_root)
            self._log_artifacts(run_id, run_root, model_family=model_family)
            self.client.set_terminated(run_id, status="FINISHED")
            return run_id
        except Exception:
            self.client.set_terminated(run_id, status="FAILED")
            raise

    def _log_parameters(self, run_id: str, config: dict[str, Any]) -> None:
        params: dict[str, Any] = {}
        training = config.get("training", {})
        if isinstance(training, dict):
            for k, v in training.items():
                if isinstance(v, (int, float, str, bool)):
                    params[f"training.{k}"] = v
        corpus = config.get("corpus", {})
        if isinstance(corpus, dict):
            for k, v in corpus.items():
                if isinstance(v, (int, float, str, bool)):
                    params[f"corpus.{k}"] = v

        for k, v in params.items():
            self.client.log_param(run_id, k, str(v))

    def _log_metrics(self, run_id: str, run_root: Path) -> None:
        metrics_csv = run_root / "metrics" / "epoch_metrics.csv"
        if not metrics_csv.exists():
            metrics_csv = run_root / "epoch_metrics.csv"

        if metrics_csv.exists():
            now_ms = int(time.time() * 1000)
            metrics_to_log: list[Metric] = []
            with open(metrics_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    step = int(row.get("epoch", 0))
                    for key, val in row.items():
                        if key == "epoch":
                            continue
                        try:
                            fval = float(val)
                            metrics_to_log.append(
                                Metric(key=key, value=fval, timestamp=now_ms, step=step)
                            )
                        except (ValueError, TypeError):
                            continue
            if metrics_to_log:
                self.client.log_batch(run_id, metrics=metrics_to_log)

    def _log_artifacts(self, run_id: str, run_root: Path, *, model_family: str) -> None:
        # Upload checkpoints, lookup artifacts, result.json, and metrics
        lookup_dir = run_root / "lookup"
        if lookup_dir.exists():
            self.client.log_artifacts(
                run_id, str(lookup_dir), artifact_path="artifacts/lookup"
            )

        ckpts_dir = run_root / "checkpoints"
        if ckpts_dir.exists():
            self.client.log_artifacts(
                run_id, str(ckpts_dir), artifact_path="artifacts/checkpoints"
            )

        res_file = run_root / "result.json"
        if res_file.exists():
            self.client.log_artifact(run_id, str(res_file), artifact_path=".")

        metrics_dir = run_root / "metrics"
        if metrics_dir.exists():
            self.client.log_artifacts(run_id, str(metrics_dir), artifact_path="metrics")


__all__ = ["PublicationService"]
