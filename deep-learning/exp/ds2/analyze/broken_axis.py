"""Full-width wave marker for DS2 broken-axis figures."""

import numpy as np
from matplotlib.lines import Line2D


def add_wave_break(figure, upper, lower) -> None:
    upper.spines.bottom.set_visible(False)
    lower.spines.top.set_visible(False)
    upper.tick_params(labelbottom=False, bottom=False)
    lower.tick_params(top=False)
    left = upper.get_position().x0
    right = upper.get_position().x1
    x = np.linspace(left, right, 800)
    phase = np.linspace(0, 24 * np.pi, len(x))
    amplitude = 0.0025
    for boundary in (upper.get_position().y0, lower.get_position().y1):
        y = boundary + amplitude * np.sin(phase)
        figure.add_artist(
            Line2D(
                x,
                y,
                transform=figure.transFigure,
                color="white",
                linewidth=5,
                solid_capstyle="round",
                clip_on=False,
                zorder=20,
            )
        )
        figure.add_artist(
            Line2D(
                x,
                y,
                transform=figure.transFigure,
                color="0.2",
                linewidth=1.2,
                clip_on=False,
                zorder=21,
            )
        )
