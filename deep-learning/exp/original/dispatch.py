"""CLI-neutral dispatch for original run and render modes."""

from __future__ import annotations

import importlib
from pathlib import Path


ORIGINAL_EXPERIMENTS = {
    "ds1": ("e01", "e02", "e03", "e04", "e05", "e06", "e09", "e10"),
    "ds2": ("e01", "e02", "e03", "e04", "e06", "e07", "e08"),
}


def select_experiments(domain: str, requested: list[str]) -> list[str]:
    available = ORIGINAL_EXPERIMENTS[domain]
    if not requested:
        return list(available)
    from exp.cli import parse_experiment_ids

    selected = parse_experiment_ids(requested)
    unknown = [item for item in selected if item not in available]
    if unknown:
        raise ValueError(
            f"experiments have no registered original trials for {domain}: "
            + ", ".join(unknown)
        )
    return selected


def run_original(
    domain: str,
    experiments: list[str],
    *,
    force: bool,
    output_dir: Path | None,
) -> None:
    module = importlib.import_module(f"exp.{domain}.original.run.api")
    root = output_dir or Path("exp") / domain / "results" / "original"
    module.run(experiments, root=root, force=force)
    importlib.import_module(f"exp.{domain}.original.render.api").render(
        experiments, root=root
    )


def analyze_original(
    domain: str,
    experiments: list[str],
    *,
    output_dir: Path | None,
) -> None:
    root = output_dir or Path("exp") / domain / "results" / "original"
    importlib.import_module(f"exp.{domain}.original.render.api").render(
        experiments, root=root
    )
