"""The single public boundary for retired DeepScratch compatibility."""

from .gateway import LegacyCompatibility
from .importer import import_legacy_archive
from .storage_audit import audit_storage, cleanup_verified_mirrors

__all__ = [
    "LegacyCompatibility",
    "audit_storage",
    "cleanup_verified_mirrors",
    "import_legacy_archive",
]
