"""Public render entrypoints for e05 language model recipes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from dlfs.identity import Variant

from .constants import (
    ADDITIONAL_BETTER_GRAPH,
    ADDITIONAL_BETTER_VALIDATION_GRAPH,
    ADDITIONAL_LSTM_GRAPH,
)
from .figures import (
    _additional_output_path,
    _default_save_figure,
    _get_collaborator,
    _render_all_recipes,
    _render_single_axis_graph,
    _single_axis_figure,
)


def render(client, error_style, output):
    del output
    include_train = getattr(client, "variant", None) is not Variant.ORIGINAL
    return _single_axis_figure(
        client,
        error_style,
        ADDITIONAL_BETTER_VALIDATION_GRAPH,
        include_train=include_train,
        evaluation_split="valid",
        evaluation_axis="epoch",
        y_min=0,
        y_max=250,
    )


def render_additional_graph(client, error_style, output) -> Path:
    """Render the previous all-recipe validation comparison separately."""
    save_fig_fn = _get_collaborator("save_figure", _default_save_figure)
    figure, _curves = _render_all_recipes(client, error_style)
    path = _additional_output_path(output, "all_rnnlm")
    save_fig_fn(figure, path)
    plt.close(figure)
    return path


def render_additional_better_graph(client, error_style, output) -> Path:
    """Render Better RNNLM with and without dropout on one regular axis."""
    return _render_single_axis_graph(
        client,
        error_style,
        output,
        ADDITIONAL_BETTER_GRAPH,
        "better_rnnlm_dropout.png",
        include_train=True,
        evaluation_split="valid",
        evaluation_axis="epoch",
        y_min=0,
        y_max=500,
    )


def render_additional_better_validation_graph(client, error_style, output) -> Path:
    """Render Better RNNLM validation perplexity as a standalone graph."""
    return _render_single_axis_graph(
        client,
        error_style,
        output,
        ADDITIONAL_BETTER_VALIDATION_GRAPH,
        "better_rnnlm_validation.png",
        evaluation_split="valid",
        evaluation_axis="epoch",
        y_min=0,
        y_max=500,
    )


def render_additional_lstm_graph(client, error_style, output) -> Path:
    """Render validation PPL for LSTM and tied LSTM RNNLMs."""
    return _render_single_axis_graph(
        client,
        error_style,
        output,
        ADDITIONAL_LSTM_GRAPH,
        "lstm_vs_tied_rnnlm.png",
    )
