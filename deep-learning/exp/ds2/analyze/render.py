"""Thin CLI dispatcher for the individual DS2 analysis modules."""

from __future__ import annotations

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
from .final_metrics import FINAL_METRIC_RENDERERS


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
SUMMARY_RENDERERS = FINAL_METRIC_RENDERERS


def _save_result(result, output):
    if isinstance(result, list):
        return result
    figure, curves = result
    save_figure(figure, output)
    write_summary(output.with_suffix(".csv"), curves)
    plt.close(figure)
    return [output, output.with_suffix(".csv")]


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
        print(f"skipping unsupported or extension analyses: {', '.join(skipped)}", file=sys.stderr)
    if not selected:
        raise ValueError("selection contains no supported analyses")
    client = AnalysisClient(
        mlflow_client(tracking_uri or tracking_uri_default()), seed=seed
    )
    root = output_dir or IMAGE_ROOT
    outputs = []
    for experiment in selected:
        seed_suffix = "" if seed is None else f"_seed-{seed}"
        if summary:
            output = root / f"{experiment}_summary{seed_suffix}.csv"
        else:
            output = root / f"{experiment}_{error_style}{seed_suffix}.png"
        outputs.extend(
            _save_result(
                renderers[experiment](client, error_style, output), output
            )
        )
    for path in outputs:
        print(path)
