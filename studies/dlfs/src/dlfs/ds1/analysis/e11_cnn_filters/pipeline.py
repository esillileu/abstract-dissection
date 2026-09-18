"""Analysis pipeline and reporting for DS1 E11 CNN filters."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from repro_core.analysis.core import mark_empty, save_figure

from ..common import runs
from .checkpoint import _checkpoint_weights_path, _conv_weights
from .plotting import (
    _panel_output,
    _render_panel,
    _shared_weight_limit,
    _write_summary,
)

RUN_GROUPS = (
    ("GT06", "CNN-SIMPLE-BOOK"),
    ("GT08", "CNN-SIMPLE-SPATIAL"),
    ("GT08", "CNN-SIMPLE-SPATIAL-PERMUTED"),
)
# Index 0 in studies/dlfs/src/dlfs/ds1/config/implemented/seeds.yaml (research_v1).
VISUALIZATION_SEED = "1"


def _visualization_runs(run_refs):
    return [run for run in run_refs if run.seed == VISUALIZATION_SEED]


def _collect(
    client, *, image: str
) -> tuple[
    list[tuple[str, str, object]],
    list[dict[str, object]],
]:
    panels: list[tuple[str, str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for group, condition in RUN_GROUPS:
        condition_runs = runs(client, group, [condition])[condition]
        for run in _visualization_runs(condition_runs):
            checkpoint = _checkpoint_weights_path(client, run)
            if checkpoint is None:
                continue
            checkpoint_path, manifest = checkpoint
            try:
                layers = _conv_weights(checkpoint_path)
            except (OSError, ValueError):
                continue
            if not layers:
                continue
            final = manifest["final"]
            checkpoint_format = str(manifest.get("format", "unknown"))
            parameter, weights = layers[0]
            panels.append((group, condition, weights))
            summary_rows.append(
                {
                    "group": group,
                    "condition": condition,
                    "seed": run.seed,
                    "run_id": run.run_id,
                    "checkpoint_format": checkpoint_format,
                    "checkpoint_epoch": final.get("epoch", ""),
                    "checkpoint_update": final.get("update", ""),
                    "parameter": parameter,
                    "shape": "x".join(str(size) for size in weights.shape),
                    "weight_min": float(weights.min()),
                    "weight_max": float(weights.max()),
                    "weight_mean": float(weights.mean()),
                    "weight_std": float(weights.std()),
                    "image": image,
                }
            )
    return panels, summary_rows


def render(client, error_style, output: Path) -> list[Path]:
    del error_style
    panels, summary_rows = _collect(client, image=output.as_posix())
    outputs = []
    if panels:
        limit = _shared_weight_limit(
            [weights for _group, _condition, weights in panels]
        )
        for panel in panels:
            group, condition, _weights = panel
            panel_output = _panel_output(output, group, condition)
            _render_panel(panel, output=panel_output, limit=limit)
            outputs.append(panel_output)
        for row in summary_rows:
            row["image"] = _panel_output(
                output, str(row["group"]), str(row["condition"])
            ).as_posix()
    else:
        figure, axis = plt.subplots(figsize=(8, 4))
        mark_empty(axis, "No completed seed-index 0 runs with final checkpoints")
        save_figure(figure, output)
        plt.close(figure)
        outputs.append(output)
    summary = output.with_suffix(".csv")
    _write_summary(summary, summary_rows)
    outputs.append(summary)
    return outputs


def render_summary(client, error_style, output: Path) -> list[Path]:
    """Write final-checkpoint filter statistics without rendering an image."""
    del error_style
    _panels, summary_rows = _collect(client, image="")
    summary = output.with_suffix(".csv")
    _write_summary(summary, summary_rows)
    for row in summary_rows:
        print(
            f"[{row['condition']}] {row['parameter']} {row['shape']}: "
            f"mean={float(row['weight_mean']):.4f}, "
            f"std={float(row['weight_std']):.4f}, "
            f"min={float(row['weight_min']):.4f}, "
            f"max={float(row['weight_max']):.4f}"
        )
    return [summary]
