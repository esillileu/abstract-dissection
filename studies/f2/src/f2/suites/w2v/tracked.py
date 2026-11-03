"""Tracked reproduction lifecycle shared by the F2 Word2Vec suites."""

from __future__ import annotations

import importlib
from pathlib import Path

from f2.suites.w2v.executor import W2VRunReceipt, _mapping
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

    # Single lifecycle evaluation (ref.md P1-4):
    # Determine lifecycle action before expensive corpus materialization.
    from f2.execution.coordinator import (
        ExecutionDecisionKind,
        F2ExecutionCoordinator,
    )
    from f2.execution.identity import ConditionIdentity

    coordinator = F2ExecutionCoordinator(tracking_uri=tracking_uri, paths=paths)
    condition = ConditionIdentity.from_config(config)
    experiment_name = str(
        _mapping(config, "tracking").get("experiment", f"f2.{condition.suite_id}")
    )
    decision = coordinator.decide(
        condition,
        experiment_name=experiment_name,
        override=is_override,
        atomic_run_id=str(config.get("atomic_run_id", "")) or None,
    )

    # 1. SKIP on finished run (no corpus materialization needed)
    if decision.kind == ExecutionDecisionKind.SKIP and not is_override:
        if progress_reporter is not None:
            progress_reporter.write(
                f"Skipping {condition.suite_id}: finished run {decision.existing_run_id} exists"
            )
        return W2VRunReceipt(
            result=None,
            run_id=decision.existing_run_id,
            staging_root=decision.run_dir,
            skipped=True,
        )

    # Resolve executor class
    exec_mod = importlib.import_module(executor_module)
    kind = str(config.get("kind", "word2vec"))
    resolver = getattr(exec_mod, "get_executor", None)
    if callable(resolver):
        executor = resolver(kind)
    else:
        executors = getattr(exec_mod, "EXECUTORS", {})
        executor = executors[kind]

    # 2. PUBLISH unposted local complete result (no corpus materialization needed)
    if decision.kind == ExecutionDecisionKind.PUBLISH and not is_override:
        if progress_reporter is not None:
            progress_reporter.write(
                f"Publishing existing local result for slot={condition.planned_run_slot_id}"
            )
        run_id = coordinator.publication.publish_run(
            experiment_name,
            condition,
            decision.run_dir,
            config,
        )
        res = getattr(executor, "_load_local_result", None)
        result = res(decision.run_dir) if callable(res) else None
        return W2VRunReceipt(
            result=result,
            run_id=run_id,
            staging_root=decision.run_dir,
            skipped=False,
        )

    # 3. Only RUN and RESUME require corpus materialization
    if progress_reporter is not None:
        progress_reporter.write("materializing verified corpus binding")
    _materialize_corpus(config, paths, progress_reporter=progress_reporter)
    if progress_reporter is not None:
        progress_reporter.write("verified corpus binding is ready")

    context = ExperimentContext(
        paths=paths,
        metadata={
            "progress_reporter": progress_reporter,
        },
    )

    return executor.execute(
        config,
        context,
        tracking_uri=tracking_uri,
        override=is_override,
        progress_reporter=progress_reporter,
        decision=decision,
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
    from f2.suites.common.w2v.corpus import slice_corpus_file

    corpus_dir = get_corpus_data_dir(paths)

    def _resolve_budgeted_corpus(source_file: Path, base_name: str) -> tuple[Path, str]:
        if "holmes" in str(source_file) or lexical_token_budget <= 0:
            from repro_io.checksum import sha256_file

            digest = corpus.get("sha256") or sha256_file(source_file)
            return source_file, str(digest)

        budget_file = corpus_dir / f"{base_name}-tokens-{lexical_token_budget}.txt"
        if progress_reporter is not None and not budget_file.is_file():
            progress_reporter.write(
                f"materializing {base_name} corpus sliced to {lexical_token_budget:,} tokens"
            )
        _, digest = slice_corpus_file(source_file, budget_file, lexical_token_budget)
        return budget_file, digest

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
        resolved_path, digest = _resolve_budgeted_corpus(corpus_path, corpus_path.stem)
        config["corpus"] = {
            "path": str(resolved_path),
            "sha256": digest,
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
            resolved_path, digest = _resolve_budgeted_corpus(candidate, corpus_id)
            config["corpus"] = {
                "path": str(resolved_path),
                "sha256": digest,
                "lexical_tokens": lexical_token_budget,
            }
            return

    from repro_io.s3 import S3ObjectStore

    from f2.corpus.db.repository import CorpusStateRepository
    from f2.corpus.db.session import get_connection
    from f2.corpus.object_store import s3_config_from_environment
    from f2.corpus.sources import SOURCE_BY_KEY
    from f2.suites.common.w2v.corpus import CorpusBinding, CorpusMaterializer

    source = SOURCE_BY_KEY.get(corpus_id)
    if source is None:
        raise FileNotFoundError(
            f"Unknown corpus {corpus_id!r}; checked {candidate_paths}"
        )
    with get_connection() as conn:
        rows = CorpusStateRepository(conn).list_verified_corpus_shards(
            source.normalized_resource_version_id
        )
    if not rows:
        raise FileNotFoundError(
            f"No verified normalized shards for {corpus_id}; prepare and validate "
            "the corpus with repro f2 corpus sources first"
        )
    materialized = CorpusMaterializer(
        S3ObjectStore(s3_config_from_environment()), paths=paths
    ).materialize(
        CorpusBinding.from_rows(rows),
        lexical_token_budget=lexical_token_budget,
        progress=(
            None
            if progress_reporter is None
            else lambda completed, total, tokens: progress_reporter.write(
                f"corpus shards processed: {completed}/{total}; "
                f"tokens: {tokens:,}/{lexical_token_budget:,}"
            )
        ),
    )
    config["corpus"] = {
        "path": str(materialized.path),
        "sha256": materialized.corpus_sha256,
        "lexical_tokens": materialized.lexical_tokens,
    }


__all__ = ["W2VRunReceipt", "run_tracked_yaml"]
