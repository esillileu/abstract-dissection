"""Word and phrase corpus resource resolution and preparation for F2 suites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from repro_core.context import RuntimePaths
from repro_io.checksum import sha256_file


def mapping(config: dict[str, object], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return dict(value)


def corpus_digest(corpus_config: dict[str, Any], source: Path) -> str:
    cfg = corpus_config.get("sha256")
    return str(cfg) if cfg is not None else sha256_file(source)


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
        from repro_io.checksum import sha256_file

        from f2.suites.w2v2.preflight import resolve_source

        resolved = resolve_source(config, paths)
        config["corpus"] = {
            "path": str(resolved),
            "sha256": corpus.get("sha256") or sha256_file(resolved),
            "lexical_tokens": corpus.get("lexical_tokens")
            or corpus.get("lexical_token_budget"),
        }
        return

    path_val = corpus.get("path")
    if path_val:
        p = Path(str(path_val))
        candidate = p if p.is_absolute() else paths.repo_root / p
        if candidate.is_file():
            config["corpus"] = {
                "path": str(candidate),
                "sha256": corpus_digest(corpus, candidate),
                "lexical_tokens": corpus.get("lexical_tokens")
                or corpus.get("lexical_token_budget"),
            }
            return

    corpus_id = str(identity.get("corpus", "")).lower()
    lexical_token_budget = int(
        corpus.get("lexical_tokens")
        or corpus.get("lexical_token_budget")
        or identity.get("budget", 0)
    )

    corpus_dir = paths.inputs("f2", "corpus")
    prepared_dir = paths.inputs("f2", "prepared")
    budget_suffix = (
        f"-{lexical_token_budget}" if lexical_token_budget and corpus_id != "holmes" else ""
    )
    candidate_paths = [
        corpus_dir / f"{corpus_id}{budget_suffix}.txt",
        corpus_dir / f"{corpus_id}.txt",
        prepared_dir / f"{corpus_id}{budget_suffix}.txt",
        prepared_dir / f"{corpus_id}.txt",
        paths.repo_root / f"workspace/inputs/f2/corpus/{corpus_id}{budget_suffix}.txt",
        paths.repo_root / f"workspace/inputs/f2/corpus/{corpus_id}.txt",
    ]
    if corpus_id == "holmes":
        from f2.suites.w2v1.resources import ensure_holmes_corpus

        candidate_paths.insert(0, ensure_holmes_corpus(corpus_dir / "holmes.txt"))

    for candidate in candidate_paths:
        if candidate.is_file():
            config["corpus"] = {
                "path": str(candidate),
                "sha256": corpus_digest(corpus, candidate),
                "lexical_tokens": lexical_token_budget,
            }
            return

    # If local candidate not found, attempt materialized corpus DB & S3 fetching
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


__all__ = ["resolve_and_prepare_corpus"]
