"""Figure construction routines for e05 language model recipes."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from repro_core.analysis.core import (
    Curve,
    mark_empty,
    plot_curve,
)
from repro_core.analysis.core import save_figure as _default_save_figure

from ..broken_axis import add_wave_break
from .constants import (
    DEFINITIONS,
    PANEL_HEIGHT_RATIOS,
    UPPER_LOG_LINEAR_THRESHOLD,
    UPPER_Y_LIMITS,
)
from .curves import (
    _evaluation_curves,
    _finite_extrema,
    _padded_limits,
)
from .curves import _train_curves as _default_train_curves
from .curves import _valid_curves as _default_valid_curves


def _get_collaborator(name: str, default):
    pkg = sys.modules.get("dlfs.ds2.analysis.e05_lm_recipes")
    if pkg is not None and hasattr(pkg, name):
        return getattr(pkg, name)
    return default


def _additional_output_path(output, graph_name: str) -> Path:
    """Keep additional graphs in the same org/imp namespace as the main one."""
    output = Path(output)
    return output.with_name(f"{output.stem}_{graph_name}{output.suffix}")


def _render_all_recipes(client, error_style):
    valid_curves_fn = _get_collaborator("_valid_curves", _default_valid_curves)
    atomic_ids = [item[0] for item in DEFINITIONS]
    curves = valid_curves_fn(client, atomic_ids)

    figure, (upper, lower) = plt.subplots(
        2,
        1,
        figsize=(9, 6),
        sharex=True,
        gridspec_kw={
            "height_ratios": PANEL_HEIGHT_RATIOS,
            "hspace": 0.05,
        },
    )
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.91)
    figure._analysis_skip_tight_layout = True
    lines = []
    for index, (atomic, label, marker, color) in enumerate(DEFINITIONS):
        axis = upper if index == 0 else lower
        line = plot_curve(
            axis,
            curves[f"{atomic}/valid"],
            label=label,
            marker=marker,
            color=color,
            error_style=error_style,
            error_every=1,
        )
        if line is not None:
            lines.append(line)
    for axis in (upper, lower):
        mark_empty(axis)
    lower_limits = _padded_limits(
        _finite_extrema([curves[f"{atomic}/valid"] for atomic in atomic_ids[1:]])
    )
    if lower_limits is not None:
        lower.set_ylim(*lower_limits)
    upper.set_yscale("symlog", linthresh=UPPER_LOG_LINEAR_THRESHOLD)
    upper.set_ylim(*UPPER_Y_LIMITS)

    lower.set_xlabel("epochs")
    figure.text(0.025, 0.5, "perplexity", va="center", rotation="vertical")
    add_wave_break(figure, upper, lower)
    if lines:
        upper.legend(handles=lines, loc="upper right")
    return figure, curves


def _single_axis_figure(
    client,
    error_style,
    definitions,
    *,
    include_train=False,
    evaluation_split="valid",
    evaluation_axis="epoch",
    y_min=None,
    y_max=None,
):
    valid_curves_fn = _get_collaborator("_valid_curves", _default_valid_curves)
    train_curves_fn = _get_collaborator("_train_curves", _default_train_curves)
    atomic_ids = [item[0] for item in definitions]
    if evaluation_split == "valid" and evaluation_axis == "epoch":
        curves = valid_curves_fn(client, atomic_ids)
    else:
        curves = _evaluation_curves(
            client, atomic_ids, split=evaluation_split, axis=evaluation_axis
        )
    train_curves = train_curves_fn(client, atomic_ids) if include_train else {}
    figure, axis = plt.subplots()
    figure._analysis_match_original_canvas = True
    for atomic, label, marker, color in definitions:
        evaluation_curve = curves[f"{atomic}/{evaluation_split}"]
        if (
            evaluation_axis == "terminal"
            and include_train
            and len(evaluation_curve.steps)
        ):
            final_step = train_curves[f"{atomic}/train"].steps.max()
            evaluation_curve = Curve(
                steps=np.full_like(evaluation_curve.steps, final_step),
                mean=evaluation_curve.mean,
                minimum=evaluation_curve.minimum,
                maximum=evaluation_curve.maximum,
                run_count=evaluation_curve.run_count,
                standard_deviation=evaluation_curve.standard_deviation,
            )
        plot_curve(
            axis,
            evaluation_curve,
            label=label,
            marker=marker,
            color=color,
            error_style=error_style,
            error_every=5,
        )
        if include_train:
            plot_curve(
                axis,
                train_curves[f"{atomic}/train"],
                label=f"{label} (train)",
                color=color,
                linestyle=":",
                error_style=error_style,
                error_every=5,
            )
    mark_empty(axis)
    axis.set(xlabel="epochs", ylabel="perplexity")
    if y_min is not None or y_max is not None:
        axis.set_ylim(bottom=y_min, top=y_max)
    if axis.has_data():
        axis.legend()
    return figure, {**curves, **train_curves}


def _render_single_axis_graph(
    client,
    error_style,
    output,
    definitions,
    filename,
    *,
    include_train=False,
    evaluation_split="valid",
    evaluation_axis="epoch",
    y_min=None,
    y_max=None,
) -> Path:
    save_fig_fn = _get_collaborator("save_figure", _default_save_figure)
    figure, _curves = _single_axis_figure(
        client,
        error_style,
        definitions,
        include_train=include_train,
        evaluation_split=evaluation_split,
        evaluation_axis=evaluation_axis,
        y_min=y_min,
        y_max=y_max,
    )
    path = _additional_output_path(output, filename.removesuffix(Path(filename).suffix))
    save_fig_fn(figure, path)
    plt.close(figure)
    return path


def _render_broken_axis_graph(
    client, error_style, output, definitions, filename
) -> Path:
    valid_curves_fn = _get_collaborator("_valid_curves", _default_valid_curves)
    save_fig_fn = _get_collaborator("save_figure", _default_save_figure)
    atomic_ids = [item[0] for item in definitions]
    curves = valid_curves_fn(client, atomic_ids)
    figure, (upper, lower) = plt.subplots(
        2,
        1,
        figsize=(6.4, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": PANEL_HEIGHT_RATIOS, "hspace": 0.05},
    )
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.91)
    figure._analysis_skip_tight_layout = True
    lines = []
    for index, (atomic, label, marker, color) in enumerate(definitions):
        axis = upper if index == 0 else lower
        line = plot_curve(
            axis,
            curves[f"{atomic}/valid"],
            label=label,
            marker=marker,
            color=color,
            error_style=error_style,
            error_every=1,
        )
        if line is not None:
            lines.append(line)
    for axis in (upper, lower):
        mark_empty(axis)
    lower_limits = _padded_limits(
        _finite_extrema([curves[f"{atomic}/valid"] for atomic in atomic_ids[1:]])
    )
    if lower_limits is not None:
        lower.set_ylim(*lower_limits)
    upper.set_yscale("symlog", linthresh=UPPER_LOG_LINEAR_THRESHOLD)
    upper.set_ylim(*UPPER_Y_LIMITS)
    lower.set_xlabel("epochs")
    figure.text(0.025, 0.5, "perplexity", va="center", rotation="vertical")
    add_wave_break(figure, upper, lower)
    if lines:
        upper.legend(handles=lines, loc="upper right")
    path = _additional_output_path(output, filename.removesuffix(Path(filename).suffix))
    save_fig_fn(figure, path)
    plt.close(figure)
    return path
