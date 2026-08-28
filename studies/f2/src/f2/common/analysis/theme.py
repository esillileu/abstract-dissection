"""Shared figure styling, Word2Vec color mappings, and plot formatting."""

from __future__ import annotations

from repro_core.plotting.theme import (
    ACCENT_COLORS,
    BACKGROUND,
    CORE_HIGHLIGHT,
    INK,
    MUTED,
    SECONDARY_DATA,
    SURFACE,
    apply_plot_theme,
)

# Standardized Word2Vec model and objective color palette
W2V_COLORS = {
    "cbow": "#356F9F",
    "skipgram": "#C66A24",
    "hierarchical_softmax": "#2F7D68",
    "negative_sampling": "#A3163B",
    "baseline": "#737D86",
    "highlight": "#A34F87",
}


def apply_f2_plot_theme() -> None:
    """Apply the repository publication plot theme."""
    apply_plot_theme()


__all__ = [
    "ACCENT_COLORS",
    "BACKGROUND",
    "CORE_HIGHLIGHT",
    "INK",
    "MUTED",
    "SECONDARY_DATA",
    "SURFACE",
    "W2V_COLORS",
    "apply_f2_plot_theme",
]
