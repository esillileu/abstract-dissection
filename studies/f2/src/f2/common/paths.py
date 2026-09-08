"""Centralized RuntimePaths helpers and storage policy for F2 study volumes."""

from __future__ import annotations

from pathlib import Path

from repro_core.context.paths import RuntimePaths


def get_f2_paths(repository_root: Path | None = None) -> RuntimePaths:
    """Return the initialized RuntimePaths for the workspace."""
    return RuntimePaths.from_environment(repository_root)


def get_corpus_data_dir(paths: RuntimePaths | None = None) -> Path:
    """Return the base materialized corpus dataset directory."""
    p = paths or get_f2_paths()
    target = p.dataset("f2") / "corpus"
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_benchmark_data_dir(paths: RuntimePaths | None = None) -> Path:
    """Return the base evaluation benchmarks dataset directory."""
    p = paths or get_f2_paths()
    target = p.dataset("f2") / "benchmarks"
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_f2_analysis_dir(
    suite: str = "corpus", paths: RuntimePaths | None = None
) -> Path:
    """Return the target deliverable directory for analysis artifacts."""
    p = paths or get_f2_paths()
    return p.analysis_output("f2", suite)


def get_f2_staging_dir(
    suite: str = "corpus", paths: RuntimePaths | None = None
) -> Path:
    """Return ephemeral scratch staging directory for a given suite."""
    p = paths or get_f2_paths()
    target = p.staging_root / "exp" / "f2" / suite
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_f2_cache_dir(paths: RuntimePaths | None = None) -> Path:
    """Return reconstructible cache root for F2 artifacts and indices."""
    p = paths or get_f2_paths()
    target = p.cache_root / "f2"
    target.mkdir(parents=True, exist_ok=True)
    return target


__all__ = [
    "get_benchmark_data_dir",
    "get_corpus_data_dir",
    "get_f2_analysis_dir",
    "get_f2_cache_dir",
    "get_f2_paths",
    "get_f2_staging_dir",
]
