"""DS1 GT01: reproduce the MNIST optimizer comparison."""

import matplotlib.pyplot as plt

from exp.framework.analysis.core import mark_empty, plot_curve

from .common import loss_curve, runs


def render(client, error_style, output):
    definitions = [
        ("MLP-OPT-SGD", "SGD", "o"),
        ("MLP-OPT-MOMENTUM", "Momentum", "x"),
        ("MLP-OPT-ADAGRAD", "AdaGrad", "s"),
        ("MLP-OPT-ADAM", "Adam", "D"),
    ]
    grouped = runs(client, "GT01", [item[0] for item in definitions])
    figure, axis = plt.subplots()
    figure._analysis_match_original_canvas = True
    curves = {}
    for atomic, label, marker in definitions:
        curve = loss_curve(client, grouped[atomic])
        curves[atomic] = curve
        plot_curve(axis, curve, label=label, marker=marker, error_style=error_style, error_every=100)
    axis.set(xlabel="iterations", ylabel="loss", ylim=(0, 1))
    mark_empty(axis)
    if axis.has_data():
        axis.legend()
    return figure, curves
