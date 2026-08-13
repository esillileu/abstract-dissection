"""Render DS2 originals without importing runners, models, or datasets."""

from __future__ import annotations

import importlib
from pathlib import Path

from exp.deepscratch.original_runtime.cache_protocol import cache_is_valid


def render(experiments: list[str], *, root: Path) -> list[Path]:
    modules = [
        (
            experiment,
            importlib.import_module(
                f"exp.deepscratch.ds2.original.native_analysis.{experiment}"
            ),
        )
        for experiment in experiments
    ]
    missing = []
    for experiment, module in modules:
        for trial_id in module.TRIAL_IDS:
            directory = root / "data" / experiment / trial_id
            if not cache_is_valid(directory):
                missing.append(f"{experiment}/{trial_id}")
    if missing:
        raise ValueError(
            "missing or invalid original trial results: " + ", ".join(missing)
        )
    image_dir = root / "image"
    outputs = []
    for _experiment, module in modules:
        outputs.extend(module.render(root, image_dir))
    print(f"rendered {len(outputs)} original figures in {image_dir}", flush=True)
    return outputs
