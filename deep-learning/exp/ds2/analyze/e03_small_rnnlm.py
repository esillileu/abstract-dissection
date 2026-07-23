"""DS2 GT03: preserve the distinct interval-loop and epoch-loop axes."""

import matplotlib.pyplot as plt

from exp.analyze import mark_empty, plot_curve

from .common import runs, source_curve


def render(client, error_style, output):
    del output
    definitions = [
        ("LM-SMALL-RNN", "interval loop", "iterations (x20)"),
        ("LM-SMALL-RNN-CUSTOM", "custom loop", "epochs"),
    ]
    grouped = runs(client, "GT03", [item[0] for item in definitions])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    curves = {}
    for axis, (atomic, title, xlabel) in zip(axes, definitions, strict=True):
        curve = source_curve(client, grouped[atomic], "perplexity")
        curves[atomic] = curve
        plot_curve(axis, curve, label="train", error_style=error_style, error_every=5)
        axis.set(title=title, xlabel=xlabel, ylabel="perplexity")
        mark_empty(axis)
        if axis.has_data():
            axis.legend()
    return figure, curves
