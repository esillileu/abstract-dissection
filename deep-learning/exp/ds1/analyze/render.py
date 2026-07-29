"""Thin CLI dispatcher for the individual DS1 analysis modules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from exp.analyze import (
    AnalysisClient,
    mlflow_client,
    parse_experiment_selection,
    save_figure,
    tracking_uri_default,
    write_summary,
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
)


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
}
SUMMARY_RENDERERS = {
    "e06": e06_summary.render,
    "e07": e07_summary.render,
    "e08": e08_summary.render,
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", "-e", action="append", default=[])
    parser.add_argument("--tracking-uri", default=tracking_uri_default())
    parser.add_argument("--error-style", choices=("band", "errorbar"), default="band")
    parser.add_argument("--output-dir", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--seed", type=int, help="actual MLflow seed/master value")
    parser.add_argument(
        "-s",
        "--summary",
        action="store_true",
        help="print final metric and training-time summaries instead of rendering figures",
    )
    args = parser.parse_args(argv)
    renderers = SUMMARY_RENDERERS if args.summary else RENDERERS
    try:
        selected, skipped = parse_experiment_selection(args.experiment, renderers)
    except ValueError as exc:
        parser.error(str(exc))
    if skipped:
        print(f"skipping unsupported or extension analyses: {', '.join(skipped)}", file=sys.stderr)
    if not selected:
        parser.error("selection contains no supported analyses")
    client = AnalysisClient(mlflow_client(args.tracking_uri), seed=args.seed)
    outputs = []
    for experiment in selected:
        output_id = experiment
        seed_suffix = "" if args.seed is None else f"_seed-{args.seed}"
        if args.summary:
            output = args.output_dir / f"{output_id}_summary{seed_suffix}.csv"
        else:
            output = args.output_dir / f"{output_id}_{args.error_style}{seed_suffix}.png"
        outputs.extend(_save_result(renderers[experiment](client, args.error_style, output), output))
    for path in outputs:
        print(path)
