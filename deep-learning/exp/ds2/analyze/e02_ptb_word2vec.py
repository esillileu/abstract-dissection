"""DS2 GT02: compare the two source PTB negative-sampling models."""

import matplotlib.pyplot as plt

from exp.analyze import mark_empty, plot_curve

from .common import runs, source_curve


def render(client, error_style, output):
    del output
    # PTB full-softmax variants are documented extension trials and stay excluded.
    definitions = [("W2V-PTB-CBOW-NS", "CBOW", "o"), ("W2V-PTB-SKIPGRAM-NS", "Skip-gram", "s")]
    grouped = runs(client, "GT02", [item[0] for item in definitions])
    figure, axis = plt.subplots(figsize=(8, 5))
    curves = {}
    for atomic, label, marker in definitions:
        curve = source_curve(client, grouped[atomic], "loss")
        curves[atomic] = curve
        plot_curve(axis, curve, label=label, marker=marker, error_style=error_style, error_every=5)
    axis.set(xlabel="iterations (x20)", ylabel="loss", title="PTB Word2Vec negative sampling")
    mark_empty(axis)
    if axis.has_data():
        axis.legend()
    return figure, curves
