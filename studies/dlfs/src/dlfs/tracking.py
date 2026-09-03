"""Canonical MLflow tracking endpoint for every DLFS workflow."""

from __future__ import annotations

import os

TRACKING_URI_ENV = "MLFLOW_F1_URL"


def tracking_uri() -> str:
    """Return the shell-provided DLFS tracking URI without generic fallbacks."""
    value = os.getenv(TRACKING_URI_ENV, "").strip()
    if not value:
        raise ValueError(
            f"{TRACKING_URI_ENV} must be loaded before running a tracked DLFS workflow"
        )
    return value.rstrip("/")


def resolve_tracking_uri(explicit: str | None = None) -> str:
    """Resolve a CLI request without overriding the shell's canonical endpoint."""
    canonical = tracking_uri()
    if explicit is not None and explicit.rstrip("/") != canonical:
        raise ValueError(
            "--tracking-uri must match the canonical URI loaded from "
            f"{TRACKING_URI_ENV}"
        )
    return canonical


__all__ = ["TRACKING_URI_ENV", "resolve_tracking_uri", "tracking_uri"]
