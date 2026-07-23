"""DS2 GT05: render the documented PTB language-model recipe analysis."""

import matplotlib.pyplot as plt
import numpy as np

from exp.analyze import aggregate, histories_from_artifact, mark_empty, plot_curve

from .common import runs, source_curve


DEFINITIONS = [
    ("LM-RNN-RECIPE", "RNNLM", "o", "tab:blue"),
    ("LM-LSTM-RECIPE", "LSTM RNNLM", "s", "tab:orange"),
    ("LM-BETTER-RECIPE", "BetterRNNLM", "D", "tab:green"),
]


def _evaluation_curve(client, run_refs, *, split, axis):
    return aggregate(
        histories_from_artifact(
            client,
            run_refs,
            artifact_path="evaluations.csv",
            x="axis_step",
            y="value",
            row_filter=lambda row: row.get("axis") == axis
            and row.get("split") == split
            and row.get("metric") == "perplexity",
            x_value=(lambda _row: 0.0) if axis == "terminal" else None,
        )
    )


def _learning_rate_curve(client, run_refs):
    return aggregate(
        histories_from_artifact(
            client,
            run_refs,
            artifact_path="updates.csv",
            x="epoch",
            y="lr",
        )
    )


def _plot_terminal(axis, curves):
    positions = []
    labels = []
    for atomic, label, _marker, color in DEFINITIONS:
        curve = curves[f"{atomic}/test"]
        if not len(curve.mean):
            continue
        position = len(positions)
        positions.append(position)
        labels.append(f"{label}\n(n={curve.run_count})")
        errors = np.asarray(
            [[curve.mean[-1] - curve.minimum[-1]], [curve.maximum[-1] - curve.mean[-1]]]
        )
        axis.bar(position, curve.mean[-1], color=color, alpha=0.75)
        axis.errorbar(position, curve.mean[-1], yerr=errors, fmt="none", ecolor="black", capsize=3)
    if positions:
        axis.set_xticks(positions, labels)
    mark_empty(axis)


def render(client, error_style, output):
    del output
    atomic_ids = [item[0] for item in DEFINITIONS]
    grouped = runs(client, "GT05", atomic_ids)
    curves = {}
    for atomic in atomic_ids:
        curves[f"{atomic}/train"] = source_curve(client, grouped[atomic], "perplexity")
        curves[f"{atomic}/valid"] = _evaluation_curve(
            client, grouped[atomic], split="valid", axis="epoch"
        )
        curves[f"{atomic}/test"] = _evaluation_curve(
            client, grouped[atomic], split="test", axis="terminal"
        )
        curves[f"{atomic}/lr"] = _learning_rate_curve(client, grouped[atomic])

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    train_axis, valid_axis, test_axis, lr_axis = axes.flat
    for atomic, label, marker, color in DEFINITIONS:
        plot_curve(
            train_axis,
            curves[f"{atomic}/train"],
            label=label,
            marker=marker,
            color=color,
            error_style=error_style,
            error_every=5,
        )
        plot_curve(
            valid_axis,
            curves[f"{atomic}/valid"],
            label=label,
            marker=marker,
            color=color,
            error_style=error_style,
            error_every=1,
        )
        plot_curve(
            lr_axis,
            curves[f"{atomic}/lr"],
            label=label,
            color=color,
            error_style=error_style,
            error_every=1,
        )
    _plot_terminal(test_axis, curves)

    train_axis.set(title="Interval train perplexity", xlabel="iterations (x20)", ylabel="perplexity")
    valid_axis.set(title="Validation perplexity", xlabel="epochs", ylabel="perplexity")
    test_axis.set(title="Selected-checkpoint test perplexity", ylabel="perplexity")
    lr_axis.set(title="Learning-rate history", xlabel="epochs", ylabel="learning rate")
    for axis in (train_axis, valid_axis, lr_axis):
        mark_empty(axis)
        if axis.has_data():
            axis.legend()
    return figure, curves
