"""DS2 GT01: compare toy CBOW and Skip-gram full-softmax loss."""

import matplotlib.pyplot as plt

from exp.analyze import mark_empty, plot_curve
from exp.plot_theme import ACCENT_COLORS

from .common import runs, source_curve


def render(client, error_style, output):
    del output
    definitions = {
        "W2V-TOY-CBOW-FULL": ("CBOW", ACCENT_COLORS[0], "-", "o"),
        # "W2V-TOY-SKIPGRAM-FULL": ("Skip-gram", ACCENT_COLORS[1], "--", "s"),
    }
    grouped = runs(client, "GT01", list(definitions))
    curves = {}
    figure, axis = plt.subplots()
    figure._analysis_match_original_canvas = True
    for atomic, (label, color, linestyle, marker) in definitions.items():
        curve = source_curve(client, grouped[atomic], "book_loss")
        curves[atomic] = curve
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
    axis.set(
        xlabel="iterations (x20)",
        ylabel="book loss",
        title="Toy Word2Vec full-softmax",
    )
    mark_empty(axis)
    if axis.has_data():
        axis.legend()
    return figure, curves
