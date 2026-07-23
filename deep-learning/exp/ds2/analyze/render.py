"""Thin CLI dispatcher for the individual DS2 analysis modules."""

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
    e01_toy_word2vec,
    e02_ptb_word2vec,
    e03_small_rnnlm,
    e04_lstm_rnnlm,
    e05_lm_recipes,
    e06_addition_seq2seq,
    e07_date_seq2seq,
    e08_attention,
)


IMAGE_ROOT = Path("exp/ds2/results/image")
RENDERERS = {
    "e01": e01_toy_word2vec.render,
    "e02": e02_ptb_word2vec.render,
    "e03": e03_small_rnnlm.render,
    "e04": e04_lstm_rnnlm.render,
    "e05": e05_lm_recipes.render,
    "e06": e06_addition_seq2seq.render,
    "e07": e07_date_seq2seq.render,
    "e08": e08_attention.render,
}


def _save_result(result, output):
    if isinstance(result, list):
        return result
    figure, curves = result
    save_figure(figure, output)
    write_summary(output.with_suffix(".csv"), curves)
    plt.close(figure)
    return [output, output.with_suffix(".csv")]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", "-e", action="append", default=[])
    parser.add_argument("--tracking-uri", default=tracking_uri_default())
    parser.add_argument("--error-style", choices=("band", "errorbar"), default="band")
    parser.add_argument("--output-dir", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--seed", type=int, help="actual MLflow seed/master value")
    args = parser.parse_args(argv)
    try:
        selected, skipped = parse_experiment_selection(args.experiment, RENDERERS)
    except ValueError as exc:
        parser.error(str(exc))
    if skipped:
        print(f"skipping unsupported or extension analyses: {', '.join(skipped)}", file=sys.stderr)
    if not selected:
        parser.error("selection contains no supported analyses")
    client = AnalysisClient(mlflow_client(args.tracking_uri), seed=args.seed)
    outputs = []
    for experiment in selected:
        seed_suffix = "" if args.seed is None else f"_seed-{args.seed}"
        output = args.output_dir / f"{experiment}_{args.error_style}{seed_suffix}.png"
        outputs.extend(_save_result(RENDERERS[experiment](client, args.error_style, output), output))
    for path in outputs:
        print(path)
