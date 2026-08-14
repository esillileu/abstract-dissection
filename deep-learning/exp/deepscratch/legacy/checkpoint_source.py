"""Resolve checkpoint dependencies from retired MLflow namespaces only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlprosection_mlflow.checkpoint_source import (
    CheckpointSourceRunNotFound,
    resolve_checkpoint_source as resolve_canonical_checkpoint_source,
    resolve_checkpoint_source_in_experiment,
)

from ..identity import Variant, Volume
from .namespaces import legacy_namespace


def resolve_checkpoint_source(
    config: dict[str, object],
    *,
    client: Any | None = None,
) -> Path | None:
    """Use canonical lookup first, then the removable legacy fallback."""
    try:
        return resolve_canonical_checkpoint_source(config, client=client)
    except CheckpointSourceRunNotFound as canonical_error:
        try:
            return resolve_legacy_checkpoint_source(config, client=client)
        except CheckpointSourceRunNotFound:
            raise canonical_error


def resolve_legacy_checkpoint_source(
    config: dict[str, object],
    *,
    client: Any | None = None,
) -> Path:
    """Resolve a DeepScratch dependency from its retired namespace.

    This is deliberately a separate fallback rather than part of canonical
    checkpoint lookup. Removing legacy storage support should require deleting
    this adapter and its single call site, without changing the normal path.
    """
    volume, variant = _coordinate(config)
    return resolve_checkpoint_source_in_experiment(
        config,
        experiment_name=legacy_namespace(volume, variant),
        client=client,
    )


def _coordinate(config: dict[str, object]) -> tuple[Volume, Variant]:
    tracking = config.get("tracking", {})
    tags = tracking.get("tags", {}) if isinstance(tracking, dict) else {}
    tags = tags if isinstance(tags, dict) else {}
    experiment = str(tracking.get("experiment", "")) if isinstance(tracking, dict) else ""
    suite = str(tags.get("suite.name", "")) or experiment.removeprefix(
        "deepscratch."
    )
    variant_name = str(tags.get("implementation.variant", ""))
    if not variant_name:
        variant_name = (
            Variant.ORIGINAL.value
            if str(config.get("kind", "")).endswith(".original")
            else Variant.IMPLEMENTED.value
        )
    try:
        return Volume(suite), Variant(variant_name)
    except ValueError as exc:
        raise CheckpointSourceRunNotFound(
            "legacy checkpoint lookup is unavailable for coordinate: "
            f"suite={suite} variant={variant_name}"
        ) from exc


__all__ = ["resolve_checkpoint_source", "resolve_legacy_checkpoint_source"]
