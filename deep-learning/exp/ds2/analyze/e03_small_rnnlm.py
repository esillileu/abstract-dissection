"""DS2 GT03: reproduce the small-corpus SimpleRnnlm perplexity graph."""

import matplotlib.pyplot as plt

from exp.analyze import mark_empty, plot_curve

from .common import runs, source_curve


def render(client, error_style, output):
    del output
    atomic = "LM-SMALL-RNN"
    grouped = runs(client, "GT03", [atomic])
    curve = source_curve(client, grouped[atomic], "perplexity")
    figure, axis = plt.subplots()
    figure._analysis_match_original_canvas = True
    plot_curve(axis, curve, label="train", error_style=error_style, error_every=5)
    axis.set(
        title="PTB small-corpus SimpleRnnlm",
        xlabel="iterations (x20)",
        ylabel="perplexity",
    )
    mark_empty(axis)
    if axis.has_data():
        axis.legend()
    return figure, {atomic: curve}
