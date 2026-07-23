"""DS2 GT06: reproduce the addition Seq2seq accuracy graph."""

import matplotlib.pyplot as plt

from exp.analyze import mark_empty, plot_curve

from .common import runs, source_curve


def render(client, error_style, output):
    del output
    definitions = [
        ("SEQA-VAN-FWD", "vanilla / forward", "o"),
        ("SEQA-VAN-REV", "vanilla / reverse", "s"),
        ("SEQA-PEEKY-FWD", "peeky / forward", "^"),
        ("SEQA-PEEKY-REV", "peeky / reverse", "D"),
    ]
    grouped = runs(client, "GT06", [item[0] for item in definitions])
    figure, axis = plt.subplots(figsize=(8, 5))
    curves = {}
    for atomic, label, marker in definitions:
        curve = source_curve(client, grouped[atomic], "exact_match_accuracy")
        curves[atomic] = curve
        plot_curve(axis, curve, label=label, marker=marker, error_style=error_style, error_every=5)
    axis.set(xlabel="epochs", ylabel="accuracy", ylim=(0, 1), title="Addition Seq2seq")
    mark_empty(axis)
    if axis.has_data():
        axis.legend()
    return figure, curves
