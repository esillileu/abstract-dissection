"""The single public boundary for retired DeepScratch compatibility."""

from .gateway import LegacyCompatibility
from .checkpoint_source import resolve_checkpoint_source, resolve_legacy_checkpoint_source
from .importer import import_legacy_archive
from .storage_audit import audit_storage, cleanup_verified_mirrors

__all__ = [
    "LegacyCompatibility",
    "audit_storage",
    "cleanup_verified_mirrors",
    "import_legacy_archive",
    "resolve_legacy_checkpoint_source",
    "resolve_checkpoint_source",
]
