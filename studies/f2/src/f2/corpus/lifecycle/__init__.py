"""Canonical non-Common-Crawl corpus lifecycle orchestration."""

from __future__ import annotations

from .acquisition import acquire_source
from .catalog import catalog_sources, install_validation_profiles
from .preflight import config_hash, git_sha, preflight, tool_versions
from .processing import (
    process_canonical_source,
    process_normalized_source,
    process_source,
)

__all__ = [
    "acquire_source",
    "catalog_sources",
    "config_hash",
    "git_sha",
    "install_validation_profiles",
    "preflight",
    "process_canonical_source",
    "process_normalized_source",
    "process_source",
    "tool_versions",
]
