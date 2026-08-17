"""DS2 GT06: reproduce the addition Seq2seq accuracy graph."""

import matplotlib.pyplot as plt

from exp.framework.analysis.core import mark_empty, plot_curve

from .common import runs, source_curve


def render(
    client,
    error_style,
    output=None,
    *,
    group_id="GT06",
    max_epoch_index=None,
    definitions=None,
    figsize=None,
    legend_loc="upper left",
):
    del output
    if definitions is None:
        definitions = (
            ("SEQA-VAN-FWD", "Vanilla-Forward", "o"),
            ("SEQA-VAN-REV", "Vanilla-Reverse", "s"),
            ("SEQA-PEEKY-FWD", "Peeky-Forward", "^"),
            ("SEQA-PEEKY-REV", "Peeky-Reverse", "D"),
        )
    grouped = runs(client, group_id, [item[0] for item in definitions])
    figure, axis = plt.subplots(figsize=figsize)
    figure._analysis_match_original_canvas = True
    curves = {}
    for atomic, label, marker in definitions:
        curve = source_curve(client, grouped[atomic], "exact_match_accuracy")
        curves[atomic] = curve
        plot_curve(axis, curve, label=label, marker=marker, error_style=error_style, error_every=5)
    axis.set(xlabel="epochs", ylabel="accuracy", ylim=(0, 1))
    if max_epoch_index is not None:
        axis.set_xlim(0, max_epoch_index)
    mark_empty(axis)
    if axis.has_data():
        axis.legend(loc=legend_loc)
    return figure, curves
