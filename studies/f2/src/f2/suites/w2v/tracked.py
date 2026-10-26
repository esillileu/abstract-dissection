"""Tracked reproduction lifecycle shared by the F2 Word2Vec suites."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from repro_io.s3 import S3ObjectStore

from f2.corpus.db.repository import CorpusStateRepository
from f2.corpus.db.session import get_connection as corpus_connection
from f2.corpus.object_store import s3_config_from_environment
from f2.suites.w2v.corpus import CorpusBinding, CorpusMaterializer
from f2.suites.w2v.executor import W2VRunReceipt, _mapping
from f2.suites.w2v.tracking import publish_run_to_mlflow
from repro_core.context import ExperimentContext, RuntimePaths


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
    override: bool = False,
    **_: object,
) -> W2VRunReceipt:
    """Execute and publish one complete paper-reproduction run."""
    if spec_module is None or executor_module is None:
        raise ValueError("tracked W2V execution requires spec and executor modules")

    parser = importlib.import_module(spec_module)
    spec = parser.parse_run_spec(path, atomic_run_id=atomic_run_id, overrides=overrides)
    spec = spec.with_seed(int(spec.identity["seed"]) if seed is None else seed)
    config = spec.to_executor_config()
    if device not in {None, "cpu"}:
        raise ValueError("canonical W2V parity execution requires CPU")

    paths = RuntimePaths.from_environment()
    is_override = (
        override
        or bool(config.get("override", False))
        or bool(overrides.get("override", False) if overrides else False)
    )

    # Skip early before expensive corpus materialization or large hash calculation
    if not is_override:
        identity = _mapping(config, "identity")
        config_digest = str(identity["config_digest"])
        seed_val = int(identity["seed"])
        suite_name = "w2v2" if "w2v2" in str(executor_module) else "w2v1"
        experiment_name = str(
            _mapping(config, "tracking").get("experiment", f"f2.{suite_name}")
        )
        from mlflow import MlflowClient

        from f2.suites.w2v.tracking import find_matching_finished_run

        client = MlflowClient(tracking_uri=tracking_uri)
        existing = find_matching_finished_run(
            client, experiment_name, config_digest, seed_val
        )
        if existing is not None:
            if progress_reporter is not None:
                progress_reporter.write(
                    f"Skipping {suite_name}: finished run {existing.info.run_id} exists"
                )
            slot_id = str(identity.get("planned_run_slot_id", ""))
            run_root = paths.runs("f2", suite_name, slot_id)
            return W2VRunReceipt(
                result=None,
                run_id=existing.info.run_id,
                staging_root=run_root,
                skipped=True,
            )

    if progress_reporter is not None:
        progress_reporter.write("materializing verified corpus binding")
    _materialize_corpus(config, paths, progress_reporter=progress_reporter)
    if progress_reporter is not None:
        progress_reporter.write("verified corpus binding is ready")

    exec_mod = importlib.import_module(executor_module)
    kind = str(config.get("kind", "word2vec"))
    resolver = getattr(exec_mod, "get_executor", None)
    if callable(resolver):
        executor = resolver(kind)
    else:
        executors = getattr(exec_mod, "EXECUTORS", {})
        executor = executors[kind]

    context = ExperimentContext(
        paths=paths,
        metadata={
            "progress_reporter": progress_reporter,
            "skip_checked": True,
        },
    )

    return executor.execute(
        config,
        context,
        tracking_uri=tracking_uri,
        override=is_override,
        progress_reporter=progress_reporter,
    )


def _materialize_corpus(
    config: dict[str, object],
    paths: RuntimePaths,
    *,
    progress_reporter: object | None = None,
) -> None:
    identity = _mapping(config, "identity")
    corpus = _mapping(config, "corpus")
    if config.get("large_scale"):
        from repro_io.checksum import sha256_file

        from f2.suites.w2v2.preflight import resolve_source

        source = resolve_source(config, paths)
        config["corpus"] = {
            **corpus,
            "path": str(source),
            "sha256": sha256_file(source),
        }
        return
    try:
        lexical_token_budget = int(corpus["lexical_token_budget"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("canonical W2V1 corpus requires lexical_token_budget") from exc
    from f2.common.paths import get_corpus_data_dir

    corpus_dir = get_corpus_data_dir(paths)
    if "path" in corpus:
        raw_path = Path(str(corpus["path"]))
        if raw_path.is_absolute():
            corpus_path = raw_path
        elif (corpus_dir / raw_path.name).is_file():
            corpus_path = corpus_dir / raw_path.name
        else:
            corpus_path = paths.repo_root / raw_path
        if "holmes" in str(raw_path):
            from f2.suites.w2v1.resources import ensure_holmes_corpus

            if progress_reporter is not None:
                progress_reporter.write(f"materializing Holmes corpus at {corpus_path}")
            corpus_path = ensure_holmes_corpus(corpus_path)
        if not corpus_path.is_file():
            raise FileNotFoundError(
                f"configured corpus path does not exist: {corpus_path}"
            )
        if "sha256" not in corpus:
            from repro_io.checksum import sha256_file

            corpus["sha256"] = sha256_file(corpus_path)
        config["corpus"] = {
            "path": str(corpus_path),
            "sha256": corpus["sha256"],
            "lexical_tokens": lexical_token_budget,
        }
        return

    atomic_run_id = str(config.get("atomic_run_id", ""))
    corpus_id = atomic_run_id.split("--", 1)[0] if "--" in atomic_run_id else ""
    candidate_paths = [
        corpus_dir / f"{corpus_id}.txt",
        corpus_dir / corpus_id / "corpus.txt",
    ]
    if corpus_id == "holmes":
        from f2.suites.w2v1.resources import ensure_holmes_corpus

        candidate_paths.insert(0, ensure_holmes_corpus(corpus_dir / "holmes.txt"))

    for candidate in candidate_paths:
        if candidate.is_file():
            from repro_io.checksum import sha256_file

            digest = corpus.get("sha256") or sha256_file(candidate)
            config["corpus"] = {
                "path": str(candidate),
                "sha256": str(digest),
                "lexical_tokens": lexical_token_budget,
            }
            return

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


def _publish(
    client: Any,
    run_id: str,
    root: Path,
    *,
    include_checkpoints: bool = False,
) -> None:
    # Forward to tracking helper if called directly
    publish_run_to_mlflow(
        client,
        "f2.w2v",
        {
            "identity": {
                "planned_run_slot_id": run_id,
                "execution_plan_id": "r1",
                "config_digest": "",
            }
        },
        root,
        include_checkpoints=include_checkpoints,
    )


__all__ = ["W2VRunReceipt", "run_tracked_yaml"]
