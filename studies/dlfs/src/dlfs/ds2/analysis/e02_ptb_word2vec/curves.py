from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from dlfs.analysis.core import write_summary
from repro_core.analysis.core import mark_empty, plot_curve, save_figure
from repro_core.plotting.theme import ACCENT_COLORS

from ..common import runs
from ..common import source_curve as _default_source_curve
from .constants import CURVE_ATOMIC_RUN_IDS


def _get_source_curve():
    """Allow monkeypatching via e02_ptb_word2vec.source_curve."""
    pkg = sys.modules.get("dlfs.ds2.analysis.e02_ptb_word2vec")
    if pkg is not None and hasattr(pkg, "source_curve"):
        return pkg.source_curve
    return _default_source_curve


def _curve_output_paths(output: Path) -> tuple[Path, Path, Path, Path]:
    """Use the same per-condition graph layout as e01."""
    stem = output.stem
    graph_stem = f"{stem}_ns"
    return (
        output.with_name(f"{graph_stem}_cbow{output.suffix}"),
        output.with_name(f"{graph_stem}_skipgram{output.suffix}"),
        output.with_name(f"{graph_stem}_combined{output.suffix}"),
        output.with_name(f"{graph_stem}_curves.csv"),
    )


def _render_ns_curves(client, error_style, output: Path) -> list[Path]:
    definitions = {
        "W2V-PTB-CBOW-NS": ("CBOW", ACCENT_COLORS[0], "-", "o"),
        "W2V-PTB-SKIPGRAM-NS": ("Skip-gram", ACCENT_COLORS[1], "--", "s"),
    }
    grouped = runs(client, "GT02", list(CURVE_ATOMIC_RUN_IDS))
    curves = {}
    outputs = []
    graph_paths = _curve_output_paths(output)
    metric = (
        "loss"
        if getattr(getattr(client, "variant", None), "value", None) == "original"
        else "book_loss"
    )
    for (atomic, (label, color, linestyle, marker)), graph_path in zip(
        definitions.items(), graph_paths[:2], strict=False
    ):
        curve = _get_source_curve()(client, grouped[atomic], metric)
        curves[label] = curve
        figure, axis = plt.subplots()
        figure._analysis_match_original_canvas = True
        plot_curve(
            axis,
            curve,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            error_style=error_style,
            error_every=5,
        )
        axis.set(xlabel="iterations (x20)", ylabel="loss")
        mark_empty(axis)
        if axis.has_data():
            axis.legend()
        save_figure(figure, graph_path)
        plt.close(figure)
        outputs.append(graph_path)
    figure_width, figure_height = plt.rcParams["figure.figsize"]
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(figure_width, figure_height * 2),
        sharex=True,
    )
    figure._analysis_match_original_canvas = True
    y_limits = {
        "CBOW": (1.3, 2.5),
        "Skip-gram": (22.0, 27.5),
    }
    for axis, (label, color, linestyle, marker) in zip(
        axes, definitions.values(), strict=False
    ):
        line = plot_curve(
            axis,
            curves[label],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            error_style=error_style,
            error_every=5,
        )
        if line is not None:
            line.set_markersize(2.5)
        axis.set(
            xlabel="iterations (x20)",
            ylabel="loss",
            ylim=y_limits[label],
        )
        mark_empty(axis)
        if axis.has_data():
            axis.legend()
    save_figure(figure, graph_paths[2])
    plt.close(figure)
    outputs.append(graph_paths[2])
    summary = graph_paths[3]
    write_summary(summary, curves)
    return [*outputs, summary]
