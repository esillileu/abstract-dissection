"""Render DS1 originals without importing any runner or upstream source."""

from __future__ import annotations

import importlib
from pathlib import Path

from exp.deepscratch.original_runtime.cache import cache_is_valid


def render(experiments: list[str], *, root: Path) -> list[Path]:
    modules = [
        (experiment, importlib.import_module(f"exp.deepscratch.ds1.original.render.{experiment}"))
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
    if "e06" in experiments:
        outputs.extend(
            importlib.import_module("exp.deepscratch.ds1.original.render.e11").render(
                root, image_dir
            )
        )
    print(f"rendered {len(outputs)} original figures in {image_dir}", flush=True)
    return outputs
