"""Publication service for publishing F2 results to MLflow.

Implements ref.md section 4.1 & 4.4:
- F2 -> MLflow atomic publication
- Querying finished runs strictly by ConditionIdentity / tags
- Native MLflow FINISHED status as sole completion criterion
- Verifies required outputs and upload integrity before marking FINISHED
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mlflow import MlflowClient
from mlflow.entities import Run
from repro_io.checksum import sha256_file

from .manifest import collect_upload_files, create_result_manifest, read_metric_batches
from .validation import validate_local_artifacts

if TYPE_CHECKING:
    from f2.execution.identity import ConditionIdentity


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

        # If condition specifies a source corpus identity, verify run has matching source_corpus_sha256
        cond_corpus_sha = condition.extra_dimensions.get("source_corpus_sha256")
        if cond_corpus_sha:
            run_corpus_sha = tags.get("f2.source_corpus_sha256")
            if run_corpus_sha and run_corpus_sha != cond_corpus_sha:
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
        """Find an existing FINISHED run matching the exact condition identity."""
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
        """Publish local execution outputs to MLflow and set FINISHED."""
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

        # 1. Pre-publication validation of local artifacts and manifest
        res_data, artifacts = validate_local_artifacts(run_root, condition)

        suite = condition.suite_id
        identity = (
            config.get("identity", {})
            if isinstance(config.get("identity"), dict)
            else {}
        )
        tags = {
            **condition.tags(),
            "suite.name": suite,
            "experiment_spec.id": str(
                identity.get("experiment_spec_id")
                or identity.get("condition_id")
                or condition.condition_id
            ),
            "implementation.variant": str(config.get("atomic_run_id", "")),
            "seed": str(condition.seed),
        }
        tracking = config.get("tracking")
        if isinstance(tracking, dict) and isinstance(tracking.get("tags"), dict):
            tags.update({str(k): str(v) for k, v in tracking["tags"].items()})
        meta = config.get("metadata")
        if isinstance(meta, dict) and isinstance(meta.get("tags"), dict):
            tags.update({str(k): str(v) for k, v in meta["tags"].items()})

        training = (
            config.get("training", {})
            if isinstance(config.get("training"), dict)
            else {}
        )
        corpus = (
            config.get("corpus", {}) if isinstance(config.get("corpus"), dict) else {}
        )
        if corpus.get("sha256"):
            tags["f2.source_corpus_sha256"] = str(corpus["sha256"])
        if corpus.get("id"):
            tags["corpus.id"] = str(corpus["id"])

        for t_k, t_v in training.items():
            if isinstance(t_v, (str, int, float, bool)):
                tags[f"training.{t_k}"] = str(t_v)

        mlflow_run = self.client.create_run(
            experiment_id=experiment_id,
            start_time=int(time.time() * 1000),
            tags=tags,
            run_name=condition.planned_run_slot_id,
        )
        run_id = mlflow_run.info.run_id

        try:
            self.client.log_param(run_id, "seed", str(condition.seed))
            for p_key, p_val in training.items():
                if isinstance(p_val, (str, int, float, bool)) and str(p_val):
                    self.client.log_param(run_id, p_key, str(p_val))
            if corpus.get("id"):
                self.client.log_param(run_id, "corpus_id", str(corpus["id"]))

            # 2. Collect and manifest upload files
            upload_files = collect_upload_files(
                run_root, artifacts, include_checkpoints=include_checkpoints
            )
            manifest_path, manifest_files = create_result_manifest(
                run_root, upload_files
            )
            upload_files.add(manifest_path)

            # 3. Upload artifacts
            for path in sorted(upload_files):
                parent = path.parent.relative_to(run_root)
                self.client.log_artifact(
                    run_id,
                    str(path),
                    artifact_path=None if parent == Path(".") else parent.as_posix(),
                )

            # 4. Verify remote artifacts were uploaded
            def list_remote_files(prefix: str | None = None) -> set[str]:
                found = set()
                for art in self.client.list_artifacts(run_id, prefix):
                    if art.is_dir:
                        found.update(list_remote_files(art.path))
                    else:
                        found.add(art.path)
                return found

            remote_files = list_remote_files()
            expected_remote = {f["path"] for f in manifest_files} | {
                "result_manifest.json"
            }
            if not expected_remote.issubset(remote_files):
                missing_remote = expected_remote - remote_files
                raise RuntimeError(
                    f"Artifact upload incomplete; missing in remote MLflow run: {missing_remote}"
                )

            # 5. Verify content integrity of essential metadata files
            with tempfile.TemporaryDirectory() as tmp_dir:
                for verify_rel in ("result.json", "result_manifest.json"):
                    if verify_rel in remote_files:
                        dl_path = self.client.download_artifacts(
                            run_id, verify_rel, tmp_dir
                        )
                        local_target = run_root / verify_rel
                        if local_target.is_file():
                            remote_hash = sha256_file(Path(dl_path))
                            local_hash = sha256_file(local_target)
                            if remote_hash != local_hash:
                                raise RuntimeError(
                                    f"Remote artifact integrity mismatch for {verify_rel}: "
                                    f"local {local_hash} != remote {remote_hash}"
                                )

            # 6. Upload observations as metrics
            metrics_file = run_root / "metrics/observations.csv"
            for batch in read_metric_batches(metrics_file):
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
