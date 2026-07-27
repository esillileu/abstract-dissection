"""Shared Matplotlib colors and styling for experiment figures."""

from __future__ import annotations

from cycler import cycler
from matplotlib import rcParams


BACKGROUND = "#F8F9F9"
INK = "#20242A"
MUTED = "#68707A"
SURFACE = "#F8F9F9"

MPL_WINTER_FATAL = (
    "#356F9F",
    "#C66A24",
    "#2F7D68",
    "#A3163B",
    "#6F4A8E",
    "#785048",
    "#A34F87",
    "#737D86",
    "#9A8A2F",
    "#2C8794",
)
ACCENT_COLORS = MPL_WINTER_FATAL

SECONDARY_DATA = "#879096"
CORE_HIGHLIGHT = "#A3163B"

FONT_FAMILY = "Times New Roman"
FONT_FALLBACKS = (
    FONT_FAMILY,
    "Times",
    "DejaVu Serif",
)
FIGURE_SIZE = (6.4, 4.8)


def apply_plot_theme() -> None:
    """Apply the repository-wide graph theme to new Matplotlib figures."""
    rcParams.update(
        {
            "axes.edgecolor": MUTED,
            "axes.facecolor": SURFACE,
            "axes.labelcolor": INK,
            "axes.prop_cycle": cycler(color=ACCENT_COLORS),
            "axes.titlecolor": INK,
            "figure.facecolor": BACKGROUND,
            "figure.figsize": FIGURE_SIZE,
            "font.family": "serif",
            "font.serif": list(FONT_FALLBACKS),
            "grid.color": MUTED,
            "legend.edgecolor": MUTED,
            "legend.facecolor": SURFACE,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": BACKGROUND,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
