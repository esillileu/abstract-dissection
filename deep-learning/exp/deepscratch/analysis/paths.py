"""Human-facing paths and names for DeepScratch analysis output."""

from __future__ import annotations

from pathlib import Path

from exp.framework.paths import WorkspacePaths

from ..identity import Variant, Volume


def default_result_root(
    volume: Volume,
    study_ids: list[str],
    variants: tuple[Variant, ...],
) -> Path:
    """Resolve derived analysis output through the workspace path policy."""
    return WorkspacePaths.from_environment(Path.cwd()).domain_results("deepscratch")


def variant_label(variants: tuple[Variant, ...]) -> str:
    if variants == (Variant.IMPLEMENTED,):
        return "imp"
    if variants == (Variant.ORIGINAL,):
        return "org"
    return "all"


def result_stem(
    volume: Volume,
    study_id: str,
    variants: tuple[Variant, ...],
) -> str:
    return f"{volume.value}_{study_id}_{variant_label(variants)}"


def selection_directory(
    root: Path,
    *,
    volume: Volume,
    study_ids: list[str],
    variants: tuple[Variant, ...],
    seed: int | None,
    run_id: str | None,
) -> Path:
    """Human-facing results are always published into one flat directory."""
    return root


__all__ = [
    "default_result_root",
    "result_stem",
    "selection_directory",
    "variant_label",
]
