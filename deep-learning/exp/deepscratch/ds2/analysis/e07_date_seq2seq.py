"""DS2 GT07: reproduce the date Seq2seq accuracy graph."""

import matplotlib.pyplot as plt

from exp.framework.analysis.core import mark_empty, plot_curve

from .common import runs, source_curve


def render(client, error_style, output):
    del output
    definitions = [
        ("SEQD-VAN-REV", "vanilla", "o"),
        ("SEQD-PEEKY-REV", "peeky", "s"),
        ("SEQD-ATTN-REV", "attention", "D"),
    ]
    grouped = runs(client, "GT07", [item[0] for item in definitions])
    figure, axis = plt.subplots()
    figure._analysis_match_original_canvas = True
    curves = {}
    for atomic, label, marker in definitions:
        curve = source_curve(client, grouped[atomic], "exact_match_accuracy")
        curves[atomic] = curve
        plot_curve(axis, curve, label=label, marker=marker, error_style=error_style, error_every=5)
    axis.set(xlabel="epochs", ylabel="accuracy", ylim=(-0.05, 1.05))
    mark_empty(axis)
    if axis.has_data():
        axis.legend()
    return figure, curves
