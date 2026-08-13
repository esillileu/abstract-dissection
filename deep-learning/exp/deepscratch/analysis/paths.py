"""Human-facing paths and names for DeepScratch analysis output."""

from __future__ import annotations

from pathlib import Path

from ..identity import Variant, Volume


RESULT_ROOT = Path(__file__).resolve().parents[1] / "results"


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
    """Keep default output flat; isolate non-default selections."""
    if seed is None and run_id is None:
        return root
    study = study_ids[0] if len(study_ids) == 1 else "all"
    suffix = f"seed-{seed}" if seed is not None else f"run-{run_id[:8]}"
    return root / f"{result_stem(volume, study, variants)}_{suffix}"


__all__ = ["RESULT_ROOT", "result_stem", "selection_directory", "variant_label"]
