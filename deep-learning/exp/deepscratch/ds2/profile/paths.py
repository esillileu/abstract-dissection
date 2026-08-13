"""Owned paths for DS2 profiling payloads and derived reports."""

from __future__ import annotations

from pathlib import Path

from exp.framework.paths import StateCoordinate, StateOwner, WorkspacePaths


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_PATHS = WorkspacePaths.from_environment(REPOSITORY_ROOT)


def profile_cache(study_id: str) -> Path:
    return WORKSPACE_PATHS.resolve(
        StateOwner.CACHE,
        StateCoordinate(
            "deepscratch", "ds2", study_id, "implemented", "profile"
        ),
    )


def profile_artifacts(study_id: str) -> Path:
    return WORKSPACE_PATHS.resolve(
        StateOwner.ARTIFACT,
        StateCoordinate(
            "deepscratch", "ds2", study_id, "implemented", "profile"
        ),
    )


__all__ = ["REPOSITORY_ROOT", "WORKSPACE_PATHS", "profile_artifacts", "profile_cache"]
