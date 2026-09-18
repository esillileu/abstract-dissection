"""Tracked two-attempt lifecycle for the canonical W2V1 vertical slice."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repro_io.s3 import S3ObjectStore

from f2.catalog.db.repository import CatalogRepository
from f2.catalog.db.session import transaction as catalog_transaction
from f2.corpus.db.repository import CorpusStateRepository
from f2.corpus.db.session import get_connection as corpus_connection
from f2.corpus.object_store import s3_config_from_environment
from f2.preflight import run_preflight
from f2.run_identity import RunIdentity
from f2.suites.w2v.corpus import CorpusBinding, CorpusMaterializer
from repro_core.context import ExperimentContext, RuntimePaths
from repro_core.execution.runner import run_config
from repro_mlflow.artifact_cache import MlflowArtifactCache
from repro_mlflow.runtime.verification import _verify_uploaded_manifest
from repro_mlflow.schema_v1 import write_result_manifest


@dataclass(frozen=True)
class TrackedW2V1Receipt:
    result: object
    run_id: str
    staging_root: Path
    durable_complete: bool


def run_tracked_yaml(
    path: str | Path,
    *,
    atomic_run_id: str | None = None,
    seed: int | None = None,
    device: str | None = None,
    overrides: dict[str, object] | None = None,
    executor_module: str | None = None,
    spec_module: str | None = None,
    tracking_uri: str,
    **_: object,
) -> TrackedW2V1Receipt:
    """Run one interruption/resume pair and publish only the final attempt."""
    if spec_module is None or executor_module is None:
        raise ValueError("tracked W2V1 execution requires spec and executor modules")
    run_preflight()
    import importlib

    from mlflow import MlflowClient

    parser = importlib.import_module(spec_module)
    config = parser.parse_run_spec(
        path, atomic_run_id=atomic_run_id, overrides=overrides
    ).to_executor_config()
    if seed is not None:
        _identity(config)["seed"] = seed
    if device not in {None, "cpu"}:
        raise ValueError("canonical W2V1 parity execution requires CPU")
    paths = RuntimePaths.from_environment()
    is_canonical = atomic_run_id != "local-smoke"
    if is_canonical:
        _materialize_corpus(config, paths)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_name = str(_mapping(config, "tracking")["experiment"])
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (
        client.create_experiment(experiment_name)
        if experiment is None
        else experiment.experiment_id
    )
    identity = _identity(config)
    first_run = _create_run(client, experiment_id, config, attempt=1)
    first_root = paths.staging_root / "exp/f2/w2v1/tracked" / first_run
    try:
        interrupted = run_config(
            config,
            ExperimentContext(
                paths=paths,
                metadata={"run_root": first_root, "stop_after_epoch": 1},
            ),
            executor_module=executor_module,
        )
        _publish(client, first_run, interrupted.root)
        client.set_tag(first_run, "trial.status", "interrupted")
        client.set_tag(first_run, "result.durable_complete", "false")
        client.set_terminated(first_run, status="KILLED")

        cache = MlflowArtifactCache(
            client, tracking_uri, root=paths.cache_root / "mlflow_artifact"
        )
        checkpoint_artifact = f"checkpoints/generations/{interrupted.checkpoint.name}"
        remote_checkpoint = cache.get(first_run, checkpoint_artifact)

        final_run = _create_run(
            client, experiment_id, config, attempt=2, predecessor=first_run
        )
        final_root = paths.staging_root / "exp/f2/w2v1/tracked" / final_run
        try:
            result = run_config(
                config,
                ExperimentContext(
                    paths=paths,
                    metadata={
                        "run_root": final_root,
                        "resume_checkpoint": remote_checkpoint,
                    },
                ),
                executor_module=executor_module,
            )
            _publish(client, final_run, result.root)
            _verify_uploaded_manifest(client, final_run)
            client.set_tag(final_run, "trial.status", "finished")
            client.set_tag(final_run, "result.durable_complete", "true")
            client.set_terminated(final_run, status="FINISHED")
            if is_canonical:
                with catalog_transaction(validate_contract=True) as connection:
                    CatalogRepository(connection).link_mlflow_run(
                        str(identity["planned_run_slot_id"]), final_run
                    )
            return TrackedW2V1Receipt(result, final_run, final_root, True)
        except BaseException:
            client.set_tag(final_run, "result.durable_complete", "false")
            client.set_tag(final_run, "trial.status", "failed")
            client.set_terminated(final_run, status="FAILED")
            raise
    except BaseException:
        run = client.get_run(first_run)
        if run.info.status == "RUNNING":
            client.set_tag(first_run, "result.durable_complete", "false")
            client.set_tag(first_run, "trial.status", "failed")
            client.set_terminated(first_run, status="FAILED")
        raise


def _materialize_corpus(config: dict[str, object], paths: RuntimePaths) -> None:
    identity = _identity(config)
    corpus = _mapping(config, "corpus")
    try:
        lexical_token_budget = int(corpus["lexical_token_budget"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("canonical W2V1 corpus requires lexical_token_budget") from exc
    version = str(identity["resource_version"])
    with corpus_connection(validate_contract=True) as connection:
        rows = CorpusStateRepository(connection).list_verified_corpus_shards(version)
    binding = CorpusBinding.from_rows(
        version, str(identity["corpus_manifest_digest"]), rows
    )
    materialized = CorpusMaterializer(
        S3ObjectStore(s3_config_from_environment()), paths=paths
    ).materialize(binding, lexical_token_budget=lexical_token_budget)
    config["corpus"] = {
        "path": str(materialized.path),
        "sha256": materialized.corpus_sha256,
        "lexical_tokens": materialized.lexical_tokens,
    }


def _create_run(
    client: Any,
    experiment_id: str,
    config: dict[str, object],
    *,
    attempt: int,
    predecessor: str | None = None,
) -> str:
    identity = _identity(config)
    run_identity = RunIdentity(
        planned_run_slot_id=str(identity["planned_run_slot_id"]),
        plan_revision=1,
        config_digest=str(identity["config_digest"]),
        resource_version_id=str(identity["resource_version"]),
        resource_manifest_digest=str(identity["corpus_manifest_digest"]),
        attempt=attempt,
        predecessor_run_id=predecessor,
    )
    tags = {
        **run_identity.tags(),
        "paper.id": "mikolov-2013-efficient-estimation",
        "suite.name": "w2v1",
        "experiment_spec.id": "w2v1-table2-cbow",
        "implementation.variant": str(config["atomic_run_id"]),
        "seed": str(identity["seed"]),
        "trial.status": "running",
        "result.durable_complete": "false",
    }
    return client.create_run(
        experiment_id,
        start_time=int(time.time() * 1000),
        tags=tags,
        run_name=f"{identity['planned_run_slot_id']}-a{attempt}",
    ).info.run_id


def _publish(client: Any, run_id: str, root: Path) -> None:
    write_result_manifest(root)
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        parent = path.parent.relative_to(root)
        client.log_artifact(
            run_id,
            str(path),
            artifact_path=None if parent == Path(".") else parent.as_posix(),
        )
    metrics = root / "metrics/observations.csv"
    if metrics.is_file():
        from mlflow.entities import Metric

        with metrics.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                step = int(row["processed_tokens"])
                client.log_batch(
                    run_id,
                    metrics=[
                        Metric(
                            "train/objective_loss",
                            float(row["objective_loss"]),
                            int(time.time() * 1000),
                            step,
                        ),
                        Metric(
                            "train/learning_rate",
                            float(row["learning_rate"]),
                            int(time.time() * 1000),
                            step,
                        ),
                        Metric(
                            "runtime/tokens_per_second",
                            float(row["tokens_per_second"]),
                            int(time.time() * 1000),
                            step,
                        ),
                    ],
                )


def _mapping(config: dict[str, object], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _identity(config: dict[str, object]) -> dict[str, Any]:
    return _mapping(config, "identity")


__all__ = ["TrackedW2V1Receipt", "run_tracked_yaml"]
