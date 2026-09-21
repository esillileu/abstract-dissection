"""Tracked reproduction lifecycle shared by the F2 Word2Vec suites."""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repro_io.s3 import S3ObjectStore

from f2.corpus.db.repository import CorpusStateRepository
from f2.corpus.db.session import get_connection as corpus_connection
from f2.corpus.object_store import s3_config_from_environment
from f2.run_identity import RunIdentity
from f2.suites.w2v.corpus import CorpusBinding, CorpusMaterializer
from repro_core.context import ExperimentContext, RuntimePaths
from repro_core.execution.runner import run_config
from repro_mlflow.runtime.verification import _verify_uploaded_manifest
from repro_mlflow.schema_v1 import write_result_manifest


@dataclass(frozen=True)
class W2VRunReceipt:
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
    progress_reporter: object | None = None,
    **_: object,
) -> W2VRunReceipt:
    """Execute and publish one complete paper-reproduction run."""
    if spec_module is None or executor_module is None:
        raise ValueError("tracked W2V execution requires spec and executor modules")
    import importlib

    from mlflow import MlflowClient

    parser = importlib.import_module(spec_module)
    spec = parser.parse_run_spec(path, atomic_run_id=atomic_run_id, overrides=overrides)
    spec = spec.with_seed(int(spec.identity["seed"]) if seed is None else seed)
    config = spec.to_executor_config()
    if device not in {None, "cpu"}:
        raise ValueError("canonical W2V parity execution requires CPU")
    paths = RuntimePaths.from_environment()
    suite = _suite(config)
    if progress_reporter is not None:
        progress_reporter.write("materializing verified corpus binding")
    _materialize_corpus(config, paths, progress_reporter=progress_reporter)
    if progress_reporter is not None:
        progress_reporter.write("verified corpus binding is ready")

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_name = str(_mapping(config, "tracking")["experiment"])
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (
        client.create_experiment(experiment_name)
        if experiment is None
        else experiment.experiment_id
    )
    run_id = _create_run(client, experiment_id, config)
    run_root = paths.staging_root / f"exp/f2/{suite}/tracked" / run_id
    try:
        result = run_config(
            config,
            ExperimentContext(
                paths=paths,
                metadata={
                    "run_root": run_root,
                    "progress_reporter": progress_reporter,
                },
            ),
            executor_module=executor_module,
        )
        _publish(client, run_id, result.root)
        _verify_uploaded_manifest(client, run_id)
        client.set_tag(run_id, "trial.status", "finished")
        client.set_tag(run_id, "result.durable_complete", "true")
        client.set_terminated(run_id, status="FINISHED")
        return W2VRunReceipt(result, run_id, run_root, True)
    except BaseException:
        run = client.get_run(run_id)
        if run.info.status == "RUNNING":
            client.set_tag(run_id, "result.durable_complete", "false")
            client.set_tag(run_id, "trial.status", "failed")
            client.set_terminated(run_id, status="FAILED")
        raise


def _materialize_corpus(
    config: dict[str, object],
    paths: RuntimePaths,
    *,
    progress_reporter: object | None = None,
) -> None:
    identity = _identity(config)
    corpus = _mapping(config, "corpus")
    try:
        lexical_token_budget = int(corpus["lexical_token_budget"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("canonical W2V1 corpus requires lexical_token_budget") from exc
    slot_id = str(identity["planned_run_slot_id"])
    with corpus_connection(validate_contract=True) as connection:
        rows = CorpusStateRepository(connection).list_verified_training_shards(slot_id)
    binding = CorpusBinding.from_rows(rows)

    def report(shard: int, total: int, tokens: int) -> None:
        if progress_reporter is not None:
            progress_reporter.write(
                f"corpus shard={shard}/{total} lexical_tokens={tokens}/{lexical_token_budget}"
            )

    materialized = CorpusMaterializer(
        S3ObjectStore(s3_config_from_environment()), paths=paths
    ).materialize(
        binding,
        lexical_token_budget=lexical_token_budget,
        progress=report,
    )
    config["corpus"] = {
        "path": str(materialized.path),
        "sha256": materialized.corpus_sha256,
        "lexical_tokens": materialized.lexical_tokens,
    }


def _create_run(
    client: Any,
    experiment_id: str,
    config: dict[str, object],
) -> str:
    identity = _identity(config)
    suite = _suite(config)
    run_identity = RunIdentity(
        planned_run_slot_id=str(identity["planned_run_slot_id"]),
        plan_revision=_plan_revision(str(identity["execution_plan_id"])),
        config_digest=str(identity["config_digest"]),
    )
    tags = {
        **run_identity.tags(),
        "paper.id": "mikolov-2013-efficient-estimation",
        "suite.name": suite,
        "experiment_spec.id": (
            "w2v1-table2-cbow"
            if suite == "w2v1"
            else "w2v2-phrase-skipgram-1b-objectives"
        ),
        "implementation.variant": str(config["atomic_run_id"]),
        "seed": str(identity["seed"]),
        "trial.status": "running",
        "result.durable_complete": "false",
    }
    return client.create_run(
        experiment_id,
        start_time=int(time.time() * 1000),
        tags=tags,
        run_name=str(identity["planned_run_slot_id"]),
    ).info.run_id


def _plan_revision(execution_plan_id: str) -> int:
    match = re.search(r"-r(\d+)$", execution_plan_id)
    if match is None:
        raise ValueError("execution_plan_id must end with an explicit revision")
    return int(match.group(1))


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


def _suite(config: dict[str, object]) -> str:
    experiment = str(_mapping(config, "tracking")["experiment"])
    suite = experiment.rsplit(".", 1)[-1]
    if suite not in {"w2v1", "w2v2"}:
        raise ValueError(f"unsupported tracked W2V suite: {suite}")
    return suite


__all__ = ["W2VRunReceipt", "run_tracked_yaml"]
