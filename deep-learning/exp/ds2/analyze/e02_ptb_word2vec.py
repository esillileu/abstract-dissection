"""DS2 GT02: compare PTB Word2Vec architecture and objective variants."""

import matplotlib.pyplot as plt

from exp.analyze import mark_empty, plot_curve

from .common import runs, source_curve


def render(client, error_style, output):
    del output
    definitions = [
        ("W2V-PTB-CBOW-NS", "CBOW", "negative sampling", "o", "-", "tab:blue"),
        ("W2V-PTB-SKIPGRAM-NS", "Skip-gram", "negative sampling", "s", "-", "tab:orange"),
        ("W2V-PTB-CBOW-FULL", "CBOW", "full softmax", "o", "--", "tab:blue"),
        ("W2V-PTB-SKIPGRAM-FULL", "Skip-gram", "full softmax", "s", "--", "tab:orange"),
    ]
    grouped = runs(client, "GT02", [item[0] for item in definitions])
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    objective_axes = dict(zip(
        ("negative sampling", "full softmax"),
        axes,
        strict=True,
    ))
    curves = {}
    for atomic, label, objective_name, marker, linestyle, color in definitions:
        curve = source_curve(client, grouped[atomic], "book_loss")
        curves[atomic] = curve
        plot_curve(
            objective_axes[objective_name],
            curve,
            label=label,
            marker=marker,
            linestyle=linestyle,
            color=color,
            error_style=error_style,
            error_every=5,
        )
    for objective_name, axis in objective_axes.items():
        axis.set(
            xlabel="iterations (x20)",
            ylabel="book loss",
            title=f"PTB Word2Vec — {objective_name}",
        )
        mark_empty(axis)
        if axis.has_data():
            axis.legend()
    figure.tight_layout()
    return figure, curves
