"""F2 Research Catalog Subsystem."""

from __future__ import annotations

from .manifest import digest_manifest, read_manifest

__all__ = [
    "digest_manifest",
    "read_manifest",
]
