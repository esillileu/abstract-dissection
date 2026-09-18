"""CLI command dispatcher for source canonical and normalized processing."""

from __future__ import annotations

from typing import Annotated

import typer

from repro_core.context.paths import RuntimePaths

from ..db.repository import CorpusStateRepository
from ..db.session import get_connection
from ..sources import SOURCE_BOUNDARY_POLICIES
from .common import _selected, _store
from .process_canonical import _process_canonical_stage, _raw_inputs
from .process_normalized import _canonical_inputs, _process_normalized_stage


def sources_process(
    source: Annotated[str, typer.Option("--source", "-s")] = "all",
    target_words: Annotated[int, typer.Option("--target-words")] = 10_000_000,
) -> None:
    """Create and publish canonical and word2vec-normalized ordered shards."""
    paths, store = RuntimePaths.from_environment(), _store()
    with get_connection() as conn:
        repo = CorpusStateRepository(conn)
        for spec in _selected(source):
            if spec.blocked_reason:
                typer.echo(f"{spec.key}: BLOCKED: {spec.blocked_reason}")
                continue
            try:
                boundary_policy = SOURCE_BOUNDARY_POLICIES.get(
                    spec.key, "sentence_per_line"
                )
                _process_canonical_stage(
                    conn, repo, spec, store, paths, target_words, boundary_policy
                )
                _process_normalized_stage(
                    conn, repo, spec, store, paths, target_words, boundary_policy
                )
            except Exception as exc:
                typer.echo(f"{spec.key}: FAILED: {exc}", err=True)


__all__ = [
    "_canonical_inputs",
    "_raw_inputs",
    "sources_process",
]
