"""DS1 GO01: reproduce four optimizer trajectories on the objective contour."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dlfs.analysis.input import histories_from_artifact
from repro_core.analysis.core import aggregate, mark_empty
from repro_core.plotting.theme import ACCENT_COLORS, CORE_HIGHLIGHT, SURFACE

from .common import runs

DEFINITIONS = (
    ("TOY-SGD", "SGD"),
    ("TOY-MOMENTUM", "Momentum"),
    ("TOY-ADAGRAD", "AdaGrad"),
    ("TOY-ADAM", "Adam"),
)
OBJECTIVE_CMAP = LinearSegmentedColormap.from_list(
    "objective_gradient",
    (
        ACCENT_COLORS[4],
        ACCENT_COLORS[4],
        ACCENT_COLORS[2],
        ACCENT_COLORS[-2],
        ACCENT_COLORS[-2],
    ),
)
OBJECTIVE_LEVELS = tuple(float(level) for level in range(8))


def _objective_contours(axis, grid_x: np.ndarray, grid_y: np.ndarray):
    objective = grid_x**2 / 20 + grid_y**2
    objective[objective > 7] = 0
    return axis.contour(
        grid_x,
        grid_y,
        objective,
        levels=OBJECTIVE_LEVELS,
        cmap=OBJECTIVE_CMAP,
        linewidths=1.5,
    )


def render(client, error_style, output):
    del output
    grouped = runs(client, "GO01", [item[0] for item in DEFINITIONS])
    figure, axes = plt.subplots(
        4,
        1,
        figsize=(4, 7),
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(left=0.1, right=0.97, bottom=0.07, top=0.97, hspace=0.16)
    figure._analysis_skip_tight_layout = True
    curves = {}
    grid_x, grid_y = np.meshgrid(np.arange(-10, 10, 0.01), np.arange(-5, 5, 0.01))
    for axis, (atomic, title) in zip(axes, DEFINITIONS, strict=True):
        _objective_contours(axis, grid_x, grid_y)
        x_curve = aggregate(
            histories_from_artifact(
                client,
                grouped[atomic],
                artifact_path="observations/trajectory.csv",
                x="update",
                y="x",
            )
        )
        y_curve = aggregate(
            histories_from_artifact(
                client,
                grouped[atomic],
                artifact_path="observations/trajectory.csv",
                x="update",
                y="y",
            )
        )
        curves[f"{atomic}/x"] = x_curve
        curves[f"{atomic}/y"] = y_curve
        if len(x_curve.steps) and np.array_equal(x_curve.steps, y_curve.steps):
            axis.plot(
                x_curve.mean,
                y_curve.mean,
                "o-",
                color=CORE_HIGHLIGHT,
                ms=4,
                label=f"mean (n={x_curve.run_count})",
                zorder=10,
            )
            if error_style == "band":
                axis.fill_between(
                    x_curve.mean,
                    y_curve.minimum,
                    y_curve.maximum,
                    color=CORE_HIGHLIGHT,
                    alpha=0.2,
                    zorder=9,
                )
            else:
                axis.errorbar(
                    x_curve.mean,
                    y_curve.mean,
                    xerr=np.vstack(
                        (x_curve.mean - x_curve.minimum, x_curve.maximum - x_curve.mean)
                    ),
                    yerr=np.vstack(
                        (y_curve.mean - y_curve.minimum, y_curve.maximum - y_curve.mean)
                    ),
                    fmt="none",
                    ecolor=CORE_HIGHLIGHT,
                    elinewidth=0.7,
                    capsize=1,
                    zorder=9,
                )
            final_x = float(x_curve.mean[-1])
            final_y = float(y_curve.mean[-1])
            axis.scatter(
                final_x,
                final_y,
                s=28,
                color=CORE_HIGHLIGHT,
                edgecolor=SURFACE,
                linewidth=0.8,
                zorder=11,
            )
            axis.annotate(
                f"({final_x:.2f}, {final_y:.2f})",
                xy=(final_x, final_y),
                xytext=(7, 6),
                textcoords="offset points",
                color=CORE_HIGHLIGHT,
                fontsize=12,
                bbox={
                    "facecolor": SURFACE,
                    "edgecolor": "none",
                    "alpha": 0.85,
                    "pad": 1,
                },
                zorder=12,
            )
        axis.plot(0, 0, "+", zorder=20)
        axis.set(xlim=(-10, 10), ylim=(-2.5, 2.5), title=title, ylabel="y")
        axis.set_aspect("equal", adjustable="box")
        if not len(x_curve.steps):
            mark_empty(axis)
    axes[-1].set_xlabel("x")
    return figure, curves
