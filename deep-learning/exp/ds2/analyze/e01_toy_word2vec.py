"""DS2 GT01: reproduce the toy CBOW loss graph."""

import matplotlib.pyplot as plt

from exp.analyze import mark_empty, plot_curve

from .common import runs, source_curve


def render(client, error_style, output):
    del output
    atomic = "W2V-TOY-CBOW-FULL"
    grouped = runs(client, "GT01", [atomic])
    curve = source_curve(client, grouped[atomic], "loss")
    figure, axis = plt.subplots(figsize=(8, 5))
    plot_curve(axis, curve, label="train", error_style=error_style, error_every=5)
    axis.set(xlabel="iterations (x20)", ylabel="loss", title="Toy CBOW full-softmax")
    mark_empty(axis)
    if axis.has_data():
        axis.legend()
    return figure, {atomic: curve}
