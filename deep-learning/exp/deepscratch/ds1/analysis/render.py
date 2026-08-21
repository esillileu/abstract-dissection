"""DS1 book-layout renderer registry."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from exp.framework.analysis.core import save_figure, write_summary

from . import (
    e01_optimizer,
    e02_initializer,
    e03_weight_decay,
    e04_dropout,
    e05_batchnorm,
    e06_simple_cnn,
    e06_e07_cnn,
    e07_deep_cnn,
    e08_spatial_layout,
    e09_optimizer_trajectory,
    e10_activation,
    e11_cnn_filters,
    e12_extended_mlp,
    e13_two_layer_net,
    e14_gradient_check,
)


RENDERERS = {
    "e01": e01_optimizer.render,
    "e02": e02_initializer.render,
    "e03": e03_weight_decay.render,
    "e04": e04_dropout.render,
    "e05": e05_batchnorm.render,
    "e06": e06_simple_cnn.render,
    "e07": e07_deep_cnn.render,
    "e08": e08_spatial_layout.render,
    "e09": e09_optimizer_trajectory.render,
    "e10": e10_activation.render,
    "e11": e11_cnn_filters.render,
    "e12": e12_extended_mlp.render,
    "e13": e13_two_layer_net.render,
    "e14": e14_gradient_check.render,
}
ADDITIONAL_RENDERERS = {
    "e07": e06_e07_cnn.render_compare,
}
STUDY_SOURCES = {
    "e07": ("e06", "e07"),
    "e11": ("e06", "e08"),
}


def render_study(
    data,
    study_id: str,
    output: Path,
    *,
    error_style: str = "band",
) -> list[Path]:
    """Render the book's study-specific composition with project error bars."""
    result = RENDERERS[study_id](data, error_style, output)
    if isinstance(result, list):
        return result
    figure, curves = result
    save_figure(figure, output)
    outputs = [output]
    additional_renderer = ADDITIONAL_RENDERERS.get(study_id)
    if additional_renderer is not None:
        compare_output = output.with_name(
            f"{output.stem}_compare{output.suffix}"
        )
        compare_figure, _compare_curves = additional_renderer(
            data, error_style, compare_output
        )
        save_figure(compare_figure, compare_output)
        plt.close(compare_figure)
        outputs.append(compare_output)
    summary = output.with_name(f"{output.stem}_curves.csv")
    write_summary(summary, curves)
    plt.close(figure)
    return [*outputs, summary]


__all__ = ["ADDITIONAL_RENDERERS", "RENDERERS", "STUDY_SOURCES", "render_study"]
