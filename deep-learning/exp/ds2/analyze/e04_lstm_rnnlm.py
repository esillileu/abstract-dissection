"""DS2 GT04: reproduce the PTB LSTM perplexity graph."""

import matplotlib.pyplot as plt

from exp.analyze import mark_empty, plot_curve

from .common import runs, source_curve


def render(client, error_style, output):
    del output
    atomic = "LM-LSTM"
    grouped = runs(client, "GT04", [atomic])
    curve = source_curve(client, grouped[atomic], "perplexity")
    figure, axis = plt.subplots(figsize=(8, 5))
    plot_curve(axis, curve, label="train", error_style=error_style, error_every=5)
    axis.set(xlabel="iterations (x20)", ylabel="perplexity", ylim=(0, 500), title="PTB LSTM RNNLM")
    mark_empty(axis)
    if axis.has_data():
        axis.legend()
    return figure, {atomic: curve}
