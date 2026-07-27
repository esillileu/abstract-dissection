"""DS1 GT02: reproduce the weight-initialization comparison."""

import matplotlib.pyplot as plt

from exp.analyze import mark_empty, plot_curve

from .common import loss_curve, runs


def render(client, error_style, output):
    del output
    definitions = [
        ("MLP-INIT-STD001", "std=0.01", "o"),
        ("MLP-INIT-XAVIER", "Xavier", "s"),
        ("MLP-INIT-HE", "He", "D"),
    ]
    grouped = runs(client, "GT02", [item[0] for item in definitions])
    figure, axis = plt.subplots()
    figure._analysis_match_original_canvas = True
    curves = {}
    for atomic, label, marker in definitions:
        curve = loss_curve(client, grouped[atomic])
        curves[atomic] = curve
        plot_curve(axis, curve, label=label, marker=marker, error_style=error_style, error_every=100)
    axis.set(xlabel="iterations", ylabel="loss", ylim=(0, 2.5), title="Weight initialization comparison")
    mark_empty(axis)
    if axis.has_data():
        axis.legend()
    return figure, curves
