"""The only registry of retired MLflow namespaces."""

from __future__ import annotations

from ..identity import Variant, Volume


LEGACY_NAMESPACES = {
    (Volume.DS1, Variant.IMPLEMENTED): "ds1",
    (Volume.DS1, Variant.ORIGINAL): "ds1_original",
    (Volume.DS2, Variant.IMPLEMENTED): "ds2",
    (Volume.DS2, Variant.ORIGINAL): "ds2_original",
}


def legacy_namespace(volume: Volume, variant: Variant) -> str:
    return LEGACY_NAMESPACES[(volume, variant)]


__all__ = ["LEGACY_NAMESPACES", "legacy_namespace"]
