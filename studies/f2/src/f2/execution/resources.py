"""Word and phrase corpus resource resolution and preparation for F2 suites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from repro_io.checksum import sha256_file

from repro_core.context import RuntimePaths


def mapping(config: dict[str, object], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return dict(value)


def resolve_and_prepare_corpus(
    config: dict[str, object],
    paths: RuntimePaths,
    *,
    progress_reporter: Any | None = None,
) -> None:
    """Ensure the required input corpus exists and bind its path and sha256 to config."""
    identity = mapping(config, "identity")
    corpus = mapping(config, "corpus")

    if config.get("large_scale"):
        from f2.suites.w2v2.preflight import resolve_source

        resolved = resolve_source(config, paths)
        config["corpus"] = {
            **corpus,
            "path": str(resolved),
            "sha256": corpus.get("sha256") or sha256_file(resolved),
            "lexical_tokens": (
                corpus.get("lexical_tokens") or corpus.get("lexical_token_budget")
            ),
        }
        return

    from f2.common.paths import get_corpus_data_dir
    from f2.suites.common.w2v.corpus import slice_corpus_file

    corpus_dir = get_corpus_data_dir(paths)

    # Determine corpus ID
    atomic_run_id = str(config.get("atomic_run_id", ""))
    corpus_id = str(identity.get("corpus") or corpus.get("id") or "").lower()
    if not corpus_id and "--" in atomic_run_id:
        corpus_id = atomic_run_id.split("--", 1)[0].lower()

    lexical_token_budget = int(
        corpus.get("lexical_tokens")
        or corpus.get("lexical_token_budget")
        or identity.get("budget", 0)
    )

    def _resolve_budgeted_corpus(source_file: Path, base_name: str) -> tuple[Path, str]:
        if corpus_id == "holmes" or lexical_token_budget <= 0:
            digest = corpus.get("sha256") or sha256_file(source_file)
            return source_file, str(digest)

        # Check if source_file is already sliced to the requested budget
        expected_suffix = f"-tokens-{lexical_token_budget}.txt"
        if source_file.name.endswith(expected_suffix):
            meta_json = source_file.with_suffix(".json")
            if meta_json.is_file():
                try:
                    meta = json.loads(meta_json.read_text(encoding="utf-8"))
                    budget_ok = (
                        int(meta.get("lexical_token_budget", -1))
                        == lexical_token_budget
                    )
                    size_ok = (
                        int(meta.get("byte_size", -1)) == source_file.stat().st_size
                    )
                    source_ok = True
                    expected_source_sha = corpus.get("sha256")
                    if expected_source_sha:
                        source_ok = (
                            meta.get("source_sha256") == expected_source_sha
                            or meta.get("corpus_sha256") == expected_source_sha
                        )
                    if (
                        budget_ok
                        and size_ok
                        and source_ok
                        and meta.get("corpus_sha256")
                    ):
                        return source_file, str(meta["corpus_sha256"])
                except Exception:
                    pass
            digest = corpus.get("sha256") or sha256_file(source_file)
            return source_file, str(digest)

        budget_file = corpus_dir / f"{base_name}-tokens-{lexical_token_budget}.txt"
        if progress_reporter is not None and not budget_file.is_file():
            progress_reporter.write(
                f"materializing {base_name} corpus sliced to {lexical_token_budget:,} tokens"
            )
        source_sha256 = corpus.get("sha256")
        _, digest = slice_corpus_file(
            source_file,
            budget_file,
            lexical_token_budget,
            source_sha256=str(source_sha256) if source_sha256 else None,
        )
        return budget_file, digest

    # 1. Configured explicit path
    path_val = corpus.get("path")
    if path_val:
        p = Path(str(path_val))
        if p.is_absolute():
            corpus_path = p
        elif (corpus_dir / p.name).is_file():
            corpus_path = corpus_dir / p.name
        else:
            corpus_path = paths.repo_root / p

        if corpus_id == "holmes":
            from f2.suites.w2v1.resources import ensure_holmes_corpus

            if progress_reporter is not None:
                progress_reporter.write(f"materializing Holmes corpus at {corpus_path}")
            corpus_path = ensure_holmes_corpus(corpus_path)

        if corpus_path.is_file():
            resolved_path, digest = _resolve_budgeted_corpus(
                corpus_path, corpus_path.stem
            )
            config["corpus"] = {
                **corpus,
                "path": str(resolved_path),
                "sha256": digest,
                "lexical_tokens": lexical_token_budget,
            }
            return
        raise FileNotFoundError(f"configured corpus path does not exist: {corpus_path}")

    # 2. Local candidate paths
    prepared_dir = paths.inputs("f2", "prepared")
    budget_suffix = (
        f"-tokens-{lexical_token_budget}"
        if lexical_token_budget and corpus_id != "holmes"
        else ""
    )
    candidate_paths: list[Path] = [
        corpus_dir / f"{corpus_id}{budget_suffix}.txt",
        corpus_dir / f"{corpus_id}.txt",
        corpus_dir / corpus_id / "corpus.txt",
        prepared_dir / f"{corpus_id}{budget_suffix}.txt",
        prepared_dir / f"{corpus_id}.txt",
    ]
    if corpus_id == "holmes":
        from f2.suites.w2v1.resources import ensure_holmes_corpus

        candidate_paths.insert(0, ensure_holmes_corpus(corpus_dir / "holmes.txt"))

    for candidate in candidate_paths:
        if candidate.is_file():
            resolved_path, digest = _resolve_budgeted_corpus(candidate, corpus_id)
            config["corpus"] = {
                **corpus,
                "path": str(resolved_path),
                "sha256": digest,
                "lexical_tokens": lexical_token_budget,
            }
            return

    # 3. If local candidate not found, attempt materialized corpus DB & S3 fetching
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
        **corpus,
        "path": str(materialized.path),
        "sha256": materialized.corpus_sha256,
        "lexical_tokens": materialized.lexical_tokens,
    }


__all__ = ["resolve_and_prepare_corpus"]
