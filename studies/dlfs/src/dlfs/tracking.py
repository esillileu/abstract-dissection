"""Canonical MLflow tracking endpoint for every DLFS workflow."""

from __future__ import annotations

import os

TRACKING_URI_ENV = "F1_MLFLOW_TRACKING_URI"


def tracking_uri() -> str:
    """Return the shell-provided DLFS tracking URI without generic fallbacks."""
    value = os.getenv(TRACKING_URI_ENV, "").strip()
    if not value:
        raise ValueError(
            f"{TRACKING_URI_ENV} must be loaded before running a tracked DLFS workflow"
        )
    return value.rstrip("/")


def resolve_tracking_uri(explicit: str | None = None) -> str:
    """Resolve an explicit CLI value before the study-owned environment value."""
    if explicit is not None:
        value = explicit.strip()
        if not value:
            raise ValueError("--tracking-uri must not be empty")
        return value.rstrip("/")
    return tracking_uri()


__all__ = ["TRACKING_URI_ENV", "resolve_tracking_uri", "tracking_uri"]
