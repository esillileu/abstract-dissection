"""Owned paths for DS2 profiling payloads and derived reports."""

from __future__ import annotations

from pathlib import Path

from exp.framework.paths import StateCoordinate, StateOwner, WorkspacePaths


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_PATHS = WorkspacePaths.from_environment(REPOSITORY_ROOT)


def _profile_cache(study_id: str, purpose: str) -> Path:
    profile_root = WORKSPACE_PATHS.resolve(
        StateOwner.CACHE,
        StateCoordinate(
            "deepscratch", "ds2", study_id, "implemented", "profile"
        ),
    )
    return profile_root / purpose


def profile_measurements(study_id: str) -> Path:
    """Return the cache owned by direct profiler measurements."""
    return _profile_cache(study_id, "measurements")


def profile_analysis(study_id: str) -> Path:
    """Return the cache owned by derived profile reports and figures."""
    return _profile_cache(study_id, "analysis")


def profile_artifacts(study_id: str) -> Path:
    """Return the cache owned by profiler-native artifacts such as nsys files."""
    return _profile_cache(study_id, "artifacts")


__all__ = [
    "REPOSITORY_ROOT",
    "WORKSPACE_PATHS",
    "profile_analysis",
    "profile_artifacts",
    "profile_measurements",
]
