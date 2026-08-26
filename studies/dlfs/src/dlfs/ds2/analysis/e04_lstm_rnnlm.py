"""DS2 GT04: reproduce the PTB LSTM perplexity graph."""

import matplotlib.pyplot as plt

from repro_core.analysis.core import mark_empty, plot_curve

from .common import runs, source_curve


def render(client, error_style, output):
    del output
    atomic = "LM-LSTM"
    grouped = runs(client, "GT04", [atomic])
    curve = source_curve(client, grouped[atomic], "perplexity")
    figure, axis = plt.subplots()
    figure._analysis_match_original_canvas = True
    plot_curve(axis, curve, label="train", error_style=error_style, error_every=5)
    axis.set(xlabel="iterations (x20)", ylabel="perplexity", ylim=(0, 500))
    mark_empty(axis)
    if axis.has_data():
        axis.legend()
    return figure, {atomic: curve}
