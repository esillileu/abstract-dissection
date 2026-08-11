"""Thin CLI dispatcher for the individual DS1 analysis modules."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from exp.analyze import (
    AnalysisClient,
    completed_seed_runs,
    mlflow_client,
    parse_experiment_selection,
    save_figure,
    tracking_uri_default,
    write_summary,
)
from exp.model_parameters import (
    append_parameter_counts,
    format_parameter_count,
    parameter_count_for_runs,
)

from . import (
    e01_optimizer,
    e02_initializer,
    e03_weight_decay,
    e04_dropout,
    e05_batchnorm,
    e06_simple_cnn,
    e06_summary,
    e07_deep_cnn,
    e07_summary,
    e08_spatial_layout,
    e08_summary,
    e09_optimizer_trajectory,
    e10_activation,
    e11_cnn_filters,
    e12_extended_mlp,
    e12_summary,
)
from .generic_summary import SUMMARY_RENDERERS as GENERIC_SUMMARY_RENDERERS


IMAGE_ROOT = Path("exp/ds1/results/image")
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
}
SUMMARY_RENDERERS = {
    **GENERIC_SUMMARY_RENDERERS,
    "e06": e06_summary.render,
    "e07": e07_summary.render,
    "e08": e08_summary.render,
    "e11": e11_cnn_filters.render_summary,
    "e12": e12_summary.render,
}
SUMMARY_MODELS = {
    "e01": (
        "GT01",
        ("MLP-OPT-SGD", "MLP-OPT-MOMENTUM", "MLP-OPT-ADAGRAD", "MLP-OPT-ADAM"),
    ),
    "e02": ("GT02", ("MLP-INIT-STD001", "MLP-INIT-XAVIER", "MLP-INIT-HE")),
    "e03": ("GT03", ("REG-WD-OFF", "REG-WD-01")),
    "e04": ("GT04", ("REG-DROPOUT-OFF", "REG-DROPOUT-ON-02")),
    "e05": (
        "GT05",
        tuple(
            f"BN-SCALE-{index:02d}-{state}"
            for index in range(1, 17)
            for state in ("ON", "OFF")
        ),
    ),
    "e06": ("GT06", e06_summary.ATOMIC_RUN_IDS),
    "e07": ("GT07", [e07_summary.ATOMIC_RUN_ID]),
    "e08": ("GT08", e08_summary.ATOMIC_RUN_IDS),
}
SUMMARY_CROSS_GROUP_MODELS = {
    "e12": e12_summary.MODELS,
}


def _save_result(result, output):
    if isinstance(result, list):
        return result
    figure, curves = result
    extra_outputs = list(getattr(figure, "_analysis_extra_outputs", ()))
    save_figure(figure, output)
    write_summary(output.with_suffix(".csv"), curves)
    plt.close(figure)
    return [output, output.with_suffix(".csv"), *extra_outputs]


def analyze(
    *,
    experiments: list[str],
    tracking_uri: str | None,
    error_style: str,
    output_dir: Path | None,
    seed: int | None,
    summary: bool,
) -> None:
    if error_style not in {"band", "errorbar"}:
        raise ValueError(f"unsupported error style: {error_style}")
    renderers = SUMMARY_RENDERERS if summary else RENDERERS
    selected, skipped = parse_experiment_selection(experiments, renderers)
    if skipped:
        print(
            f"skipping unsupported or extension analyses: {', '.join(skipped)}",
            file=sys.stderr,
        )
    if not selected:
        raise ValueError("selection contains no supported analyses")
    client = AnalysisClient(
        mlflow_client(tracking_uri or tracking_uri_default()), seed=seed
    )
    root = output_dir or IMAGE_ROOT
    outputs = []
    for experiment in selected:
        output_id = experiment
        seed_suffix = "" if seed is None else f"_seed-{seed}"
        if summary:
            output = root / f"{output_id}_summary{seed_suffix}.csv"
        else:
            output = root / f"{output_id}_{error_style}{seed_suffix}.png"
        outputs.extend(
            _save_result(renderers[experiment](client, error_style, output), output)
        )
        if summary and experiment in SUMMARY_MODELS:
            group_id, atomic_run_ids = SUMMARY_MODELS[experiment]
            grouped_runs = completed_seed_runs(
                client,
                experiment_name="ds1",
                group_id=group_id,
                atomic_run_ids=atomic_run_ids,
            )
            parameter_counts = {
                atomic_run_id: parameter_count_for_runs(
                    client,
                    grouped_runs[atomic_run_id],
                )
                for atomic_run_id in atomic_run_ids
            }
            for atomic_run_id, parameter_count in parameter_counts.items():
                print(f"[{atomic_run_id}] {format_parameter_count(parameter_count)}")
            append_parameter_counts(output.with_suffix(".csv"), parameter_counts)
        if summary and experiment in SUMMARY_CROSS_GROUP_MODELS:
            parameter_counts = {}
            for group_id, atomic_run_id in SUMMARY_CROSS_GROUP_MODELS[experiment]:
                grouped_runs = completed_seed_runs(
                    client,
                    experiment_name="ds1",
                    group_id=group_id,
                    atomic_run_ids=[atomic_run_id],
                )
                parameter_counts[atomic_run_id] = parameter_count_for_runs(
                    client,
                    grouped_runs[atomic_run_id],
                )
            for atomic_run_id, parameter_count in parameter_counts.items():
                print(f"[{atomic_run_id}] {format_parameter_count(parameter_count)}")
            append_parameter_counts(output.with_suffix(".csv"), parameter_counts)
    for path in outputs:
        print(path)
