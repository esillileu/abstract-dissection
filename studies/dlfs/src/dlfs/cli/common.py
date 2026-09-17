"""Shared CLI helper utilities for DLFS commands."""

from __future__ import annotations

import json

from dlfs.identity import Variant, Volume


def _selected_variant(variant: Variant, original: bool) -> Variant:
    if original and variant not in {Variant.IMPLEMENTED, Variant.ORIGINAL}:
        raise ValueError("-o cannot be combined with this --variant")
    return Variant.ORIGINAL if original else variant


def _writer_overrides(
    volume: Volume,
    variant: Variant,
    values: list[str],
) -> list[str]:
    schema = f"{volume.value}-{variant.value}"
    tags = {
        "domain.name": "deepscratch",
        "suite.name": volume.value,
        "deepscratch.volume": volume.value,
        "implementation.variant": variant.value,
        "experiment.id": "{experiment_id}",
        "condition.id": "{condition_id}",
        "result.schema.name": schema,
        "result.schema.version": "1",
    }
    return [
        *values,
        f"tracking.experiment=deepscratch.{volume.value}",
        f"tracking.tags={json.dumps(tags, separators=(',', ':'))}",
    ]
