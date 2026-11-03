"""Publication service for publishing F2 results to MLflow.

Implements ref.md section 4.1 & 4.4:
- F2 -> MLflow atomic publication
- Querying finished runs strictly by ConditionIdentity / tags
- Native MLflow FINISHED status as sole completion criterion
- Verifies required outputs and upload integrity before marking FINISHED
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from mlflow import MlflowClient
from mlflow.entities import Metric, Run

from f2.execution.identity import ConditionIdentity


def _extract_plan_revision(execution_plan_id: str) -> int:
    match = re.search(r"-r(\d+)$", execution_plan_id)
    if match is None:
        return 1
    return int(match.group(1))


class PublicationService:
    """Manages publication of verified local run artifacts to MLflow."""

    def __init__(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri=tracking_uri)

    def get_finished_runs_for_experiment(self, experiment_name: str) -> list[Run]:
        """Fetch all FINISHED runs for an experiment with pagination."""
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return []

        all_runs: list[Run] = []
        page_token = None
        while True:
            page = self.client.search_runs(
                [experiment.experiment_id],
                filter_string="attributes.status = 'FINISHED'",
                max_results=1000,
                page_token=page_token,
            )
            all_runs.extend(page)
            page_token = getattr(page, "token", None)
            if not page_token:
                break
        return all_runs

    def run_matches_condition(
        self,
        run: Run,
        condition: ConditionIdentity,
        *,
        atomic_run_id: str | None = None,
    ) -> bool:
        """Check if a MLflow run strictly matches the condition identity."""
        tags = run.data.tags
        digest = tags.get("f2.config_digest") or tags.get("config_digest")
        run_seed = tags.get("f2.seed") or tags.get("seed")

        if not digest or digest != condition.config_digest:
            return False
        if not run_seed or run_seed != str(condition.seed):
            return False

        if condition.planned_run_slot_id:
            slot = tags.get("f2.planned_run_slot_id") or tags.get("planned_run_slot_id")
            if not slot or slot != condition.planned_run_slot_id:
                return False

        if atomic_run_id:
            run_variant = (
                tags.get("implementation.variant")
                or tags.get("atomic_run_id")
                or tags.get("f2.atomic_run_id")
            )
            if not run_variant or run_variant != atomic_run_id:
                return False

        return True

    def find_matching_finished_run(
        self,
        experiment_name: str,
        condition: ConditionIdentity,
        *,
        atomic_run_id: str | None = None,
        cached_runs: list[Run] | None = None,
    ) -> Run | None:
        """Find an existing FINISHED run matching the exact condition identity.

        If cached_runs is provided, searches locally without making remote MLflow calls.
        """
        runs = (
            cached_runs
            if cached_runs is not None
            else self.get_finished_runs_for_experiment(experiment_name)
        )
        for run in runs:
            if self.run_matches_condition(run, condition, atomic_run_id=atomic_run_id):
                return run
        return None

    def publish_run(
        self,
        experiment_name: str,
        condition: ConditionIdentity,
        run_root: Path,
        config: dict[str, Any],
        *,
        include_checkpoints: bool | None = None,
    ) -> str:
        """Publish local execution outputs to MLflow and set FINISHED.

        Marks the run FINISHED only after upload completes successfully.
        If an upload error occurs, the run is terminated as FAILED and the error is raised.
        Local files are always preserved.
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = self.client.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        if include_checkpoints is None:
            include_checkpoints = bool(config.get("large_scale")) or (
                isinstance(config.get("training"), dict)
                and config["training"].get("model_kind") == "feedforward_nnlm"
            )

        # Pre-publication validation (ref.md P0-2):
        # Validate result.json, complete=True, identity match, and required outputs before creating MLflow run.
        result_file = run_root / "result.json"
        if not result_file.is_file():
            raise ValueError(f"Cannot publish run: missing result.json in {run_root}")

        try:
            res_data = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(
                f"Cannot publish run: corrupted result.json: {exc}"
            ) from exc

        if not res_data.get("complete"):
            raise ValueError(
                f"Cannot publish incomplete run: complete={res_data.get('complete')}"
            )

        res_identity = res_data.get("identity")
        if isinstance(res_identity, dict):
            res_digest = res_identity.get("config_digest")
            if res_digest and res_digest != condition.config_digest:
                raise ValueError(
                    f"Cannot publish run: config_digest mismatch ({res_digest} != {condition.config_digest})"
                )
            res_seed = res_identity.get("seed")
            if res_seed is not None and int(res_seed) != condition.seed:
                raise ValueError(
                    f"Cannot publish run: seed mismatch ({res_seed} != {condition.seed})"
                )
            res_slot = res_identity.get("planned_run_slot_id")
            if (
                res_slot
                and condition.planned_run_slot_id
                and res_slot != condition.planned_run_slot_id
            ):
                raise ValueError(
                    f"Cannot publish run: slot_id mismatch ({res_slot} != {condition.planned_run_slot_id})"
                )

        artifacts = res_data.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError(
                "Cannot publish run: missing artifacts dictionary in result.json"
            )

        lookup_val = artifacts.get("lookup")
        if not lookup_val:
            raise ValueError("Cannot publish run: missing lookup in artifacts")
        final_lookup_path = run_root / lookup_val
        if not final_lookup_path.exists():
            raise ValueError(
                f"Cannot publish run: designated lookup does not exist: {final_lookup_path}"
            )

        checkpoint_val = artifacts.get("checkpoint")
        if checkpoint_val:
            chk_path = run_root / checkpoint_val
            if not chk_path.exists():
                raise ValueError(
                    f"Cannot publish run: designated checkpoint does not exist: {chk_path}"
                )

        metrics_val = artifacts.get("metrics")
        if metrics_val:
            met_path = run_root / metrics_val
            if not met_path.exists():
                raise ValueError(
                    f"Cannot publish run: designated metrics file does not exist: {met_path}"
                )

        suite = condition.suite_id
        identity = (
            config.get("identity", {})
            if isinstance(config.get("identity"), dict)
            else {}
        )
        training = (
            config.get("training", {})
            if isinstance(config.get("training"), dict)
            else {}
        )
        corpus = (
            config.get("corpus", {}) if isinstance(config.get("corpus"), dict) else {}
        )
        model_kind = str(training.get("model_kind", ""))
        corpus_id = str(corpus.get("id", corpus.get("name", "")))
        is_phrase = bool(
            config.get("phrase", False)
            or "phrase" in str(config.get("atomic_run_id", ""))
        )
        objective_kind = str(training.get("objective_kind", ""))
        dimension = str(
            training.get("embedding_dimension")
            or training.get("hidden_size")
            or training.get("dim", "")
        )
        window_radius = str(
            training.get("window_radius") or training.get("window_size", "")
        )
        subsampling = str(training.get("subsampling_threshold", ""))
        negative_sample_count = str(
            training.get("negative_sample_count")
            or training.get("negative_samples", "")
        )
        epochs = str(training.get("epochs") or training.get("max_epochs", ""))

        tags = {
            **condition.tags(),
            "f2.source_corpus_sha256": str(corpus.get("sha256", "")),
            "paper.id": (
                "mikolov-2013-efficient-estimation"
                if suite == "w2v1"
                else "mikolov-2013-distributed-representations"
            ),
            "suite.name": suite,
            "experiment_spec.id": str(
                identity.get(
                    "experiment_spec_id",
                    (
                        "w2v1-table2-cbow"
                        if suite == "w2v1"
                        else "w2v2-phrase-skipgram-1b-objectives"
                    ),
                )
            ),
            "implementation.variant": str(config.get("atomic_run_id", "")),
            "seed": str(condition.seed),
            "training.model_kind": model_kind,
            "corpus.id": corpus_id,
            "is_phrase": str(is_phrase),
            "training.objective": objective_kind,
            "training.dim": dimension,
            "training.window": window_radius,
            "training.subsample": subsampling,
            "training.negative_samples": negative_sample_count,
            "training.epochs": epochs,
        }

        mlflow_run = self.client.create_run(
            experiment_id=experiment_id,
            start_time=int(time.time() * 1000),
            tags=tags,
            run_name=condition.planned_run_slot_id,
        )
        run_id = mlflow_run.info.run_id

        try:
            # Log params using actual configuration keys and query aliases
            for p_key, p_val in {
                "model_kind": model_kind,
                "corpus_id": corpus_id,
                "objective_kind": objective_kind,
                "embedding_dimension": dimension,
                "window_radius": window_radius,
                "subsampling_threshold": subsampling,
                "negative_sample_count": negative_sample_count,
                "dim": dimension,
                "window_size": window_radius,
                "negative_samples": negative_sample_count,
                "epochs": epochs,
                "seed": str(condition.seed),
            }.items():
                if p_val:
                    self.client.log_param(run_id, p_key, p_val)

            # 1. Identify designated final model path from result.json
            final_lookup_rel: Path | None = Path(lookup_val)

            def should_upload(file_path: Path) -> bool:
                rel = file_path.relative_to(run_root)
                if "preparation" in rel.parts:
                    return False
                # Intermediate pointers (latest.json, best.json, etc.) and hidden/tmp files remain local only
                if (
                    file_path.name in {"latest.json", "best.json"}
                    or file_path.name.startswith(".")
                    or file_path.name.endswith(".tmp")
                ):
                    return False
                # Always upload the designated final model
                if final_lookup_rel is not None and (
                    rel == final_lookup_rel or final_lookup_rel in rel.parents
                ):
                    return True
                # Intermediate checkpoints (generations or periodic/epoch checkpoints) remain local only
                if "generations" in rel.parts or any(
                    part.startswith("latest-epoch-")
                    or part.startswith("periodic-")
                    or part.startswith("best-")
                    or part.startswith("epoch-")
                    for part in rel.parts
                ):
                    return False
                if not include_checkpoints and "checkpoints" in rel.parts:
                    return False
                return True

            # 2. Write local result manifest matching exactly what is uploaded
            manifest_path = run_root / "result_manifest.json"
            manifest_files = []
            for path in sorted(item for item in run_root.rglob("*") if item.is_file()):
                if path == manifest_path or not should_upload(path):
                    continue
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                manifest_files.append(
                    {
                        "path": path.relative_to(run_root).as_posix(),
                        "size": path.stat().st_size,
                        "sha256": digest.hexdigest(),
                    }
                )
            manifest_payload = {"files": manifest_files, "checkpoints": []}
            manifest_path.write_text(
                json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            # 3. Upload artifacts matching result manifest
            for path in sorted(item for item in run_root.rglob("*") if item.is_file()):
                if not should_upload(path) and path != manifest_path:
                    continue
                parent = path.parent.relative_to(run_root)
                self.client.log_artifact(
                    run_id,
                    str(path),
                    artifact_path=None if parent == Path(".") else parent.as_posix(),
                )

            # 4. Upload observations as metrics
            metrics_file = run_root / "metrics/observations.csv"
            if metrics_file.is_file():
                batch: list[Metric] = []
                now_ms = int(time.time() * 1000)
                with metrics_file.open(encoding="utf-8", newline="") as stream:
                    for row in csv.DictReader(stream):
                        step = int(row.get("processed_tokens", 0))
                        if "objective_loss" in row:
                            batch.append(
                                Metric(
                                    "train/objective_loss",
                                    float(row["objective_loss"]),
                                    now_ms,
                                    step,
                                )
                            )
                        if "learning_rate" in row:
                            batch.append(
                                Metric(
                                    "train/learning_rate",
                                    float(row["learning_rate"]),
                                    now_ms,
                                    step,
                                )
                            )
                        if "tokens_per_second" in row:
                            batch.append(
                                Metric(
                                    "runtime/tokens_per_second",
                                    float(row["tokens_per_second"]),
                                    now_ms,
                                    step,
                                )
                            )
                        if len(batch) >= 750:
                            self.client.log_batch(run_id, metrics=batch)
                            batch = []
                    if batch:
                        self.client.log_batch(run_id, metrics=batch)

            # Mark as FINISHED only after complete upload
            self.client.set_terminated(run_id, status="FINISHED")
            return run_id
        except BaseException:
            try:
                self.client.set_terminated(run_id, status="FAILED")
            except Exception:
                pass
            raise


__all__ = ["PublicationService"]
