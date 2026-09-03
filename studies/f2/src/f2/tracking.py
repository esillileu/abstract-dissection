"""Canonical MLflow tracking endpoint contract for F2 workflows."""

from __future__ import annotations

import os

TRACKING_URI_ENV = "F2_MLFLOW_TRACKING_URI"


def resolve_tracking_uri(explicit: str | None = None) -> str:
    """Resolve an explicit value before the F2-owned environment value."""
    value = (
        explicit.strip()
        if explicit is not None
        else os.getenv(TRACKING_URI_ENV, "").strip()
    )
    if not value:
        source = "--tracking-uri" if explicit is not None else TRACKING_URI_ENV
        raise ValueError(f"{source} must provide an F2 MLflow tracking URI")
    return value.rstrip("/")


__all__ = ["TRACKING_URI_ENV", "resolve_tracking_uri"]
