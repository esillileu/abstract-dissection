"""Tracking-neutral curve aggregation and plotting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from repro_core.plotting.theme import MUTED, apply_plot_theme, remove_figure_title

apply_plot_theme()


@dataclass(frozen=True)
class Curve:
    steps: np.ndarray
    mean: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    run_count: int
    standard_deviation: np.ndarray | None = None

    @classmethod
    def empty(cls) -> Curve:
        empty = np.asarray([], dtype=float)
        return cls(empty, empty, empty, empty, 0, empty)


def aggregate(histories: Sequence[Mapping[float, float]]) -> Curve:
    """Aggregate only x coordinates shared by every available seed history."""
    if not histories:
        return Curve.empty()
    common_steps = set(histories[0])
    for history in histories[1:]:
        common_steps.intersection_update(history)
    if not common_steps:
        return Curve.empty()
    steps = np.asarray(sorted(common_steps), dtype=float)
    values = np.asarray(
        [[history[step] for step in steps] for history in histories], dtype=float
    )
    standard_deviation = (
        values.std(axis=0, ddof=1) if len(histories) > 1 else np.zeros_like(steps)
    )
    return Curve(
        steps=steps,
        mean=values.mean(axis=0),
        minimum=values.min(axis=0),
        maximum=values.max(axis=0),
        run_count=len(histories),
        standard_deviation=standard_deviation,
    )


def plot_curve(
    axis,
    curve: Curve,
    *,
    label: str,
    error_style: str,
    marker: str | None = None,
    linestyle: str = "-",
    error_every: int = 5,
    color: str | None = None,
):
    if not len(curve.steps):
        return None
    line = axis.plot(
        curve.steps,
        curve.mean,
        label=f"{label} (n={curve.run_count})",
        marker=marker,
        markevery=max(1, error_every),
        markersize=5,
        linestyle=linestyle,
        linewidth=1.6,
        color=color,
    )[0]
    if error_style == "band":
        if curve.standard_deviation is None:
            lower, upper = curve.minimum, curve.maximum
        else:
            lower = curve.mean - curve.standard_deviation
            upper = curve.mean + curve.standard_deviation
        axis.fill_between(
            curve.steps,
            lower,
            upper,
            color=line.get_color(),
            alpha=0.2,
            linewidth=0,
        )
    elif error_style == "errorbar":
        errors = np.maximum(
            0.0,
            np.vstack((curve.mean - curve.minimum, curve.maximum - curve.mean)),
        )
        axis.errorbar(
            curve.steps,
            curve.mean,
            yerr=errors,
            fmt="none",
            ecolor=line.get_color(),
            errorevery=max(1, error_every),
            elinewidth=0.8,
            capsize=1.5,
        )
    else:
        raise ValueError(f"unknown error style: {error_style}")
    return line


def mark_empty(axis, message: str = "No completed runs") -> None:
    if axis.has_data():
        return
    axis.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=axis.transAxes,
        color=MUTED,
    )


def save_figure(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    remove_figure_title(figure)
    match_original = getattr(figure, "_analysis_match_original_canvas", False)
    if not match_original and not getattr(figure, "_analysis_skip_tight_layout", False):
        figure.tight_layout()
    if match_original:
        figure.savefig(path, dpi=figure.dpi)
    else:
        figure.savefig(path, dpi=160, bbox_inches="tight")
    return path
