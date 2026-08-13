"""The single public boundary for retired DeepScratch compatibility."""

from .fixed_seed import fixed_seed_root
from .gateway import LegacyCompatibility
from .importer import import_legacy_archive
from .storage_audit import audit_storage, cleanup_verified_mirrors

__all__ = [
    "LegacyCompatibility",
    "audit_storage",
    "cleanup_verified_mirrors",
    "fixed_seed_root",
    "import_legacy_archive",
]
