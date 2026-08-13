"""Read-only location of quarantined fixed-seed payloads."""

from __future__ import annotations

from pathlib import Path

from exp.framework.paths import WorkspacePaths

from ..identity import Variant, Volume
from .namespaces import legacy_namespace


def fixed_seed_root(
    volume: Volume,
    *,
    repository_root: Path | None = None,
) -> Path:
    """Resolve quarantined data, with a read-only source-tree migration fallback."""
    root = (repository_root or Path.cwd()).resolve()
    canonical = (
        WorkspacePaths.from_environment(root).legacy_root
        / legacy_namespace(volume, Variant.ORIGINAL)
        / "fixed_seed"
    )
    if canonical.exists():
        return canonical
    return root / f"exp/deepscratch/{volume.value}/original/legacy_results/fixed_seed"
