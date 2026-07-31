"""Summaries of DS2 original-code result caches."""

from __future__ import annotations

from pathlib import Path

from exp.original.summary import (
    OriginalMetric,
    last_value,
    write_experiment_summary,
)


SUPPORTED_EXPERIMENTS = ("e01", "e02", "e03", "e04", "e06", "e07")
TRIAL_IDS = {
    "e01": ("dlfs2.ch03.toy-cbow-full-softmax",),
    "e02": tuple(
        (
            f"dlfs2.ch04.ptb-{name}-negative-sampling"
            if objective == "negative-sampling"
            else f"ext.ds2.ptb-{name}-{objective}"
        )
        for name in ("cbow", "skipgram")
        for objective in (
            "negative-sampling",
            "full-softmax",
            "onehot-full-softmax",
        )
    ),
    "e03": ("dlfs2.ch05.ptb-small-rnnlm",),
    "e04": ("dlfs2.ch06.ptb-lstm-rnnlm",),
    "e06": tuple(
        f"dlfs2.ch07.addition.{name}"
        for name in (
            "seq2seq-forward",
            "seq2seq-reverse",
            "peeky-seq2seq-forward",
            "peeky-seq2seq-reverse",
        )
    ),
    "e07": tuple(
        f"dlfs2.ch08.date.{name}"
        for name in (
            "seq2seq-reverse",
            "peeky-seq2seq-reverse",
            "attention-seq2seq-reverse",
        )
    ),
}


def _metrics(experiment: str, rows: list[dict[str, str]]) -> list[OriginalMetric]:
    if experiment in {"e01", "e02"}:
        value = last_value(rows, "loss")
        return [] if value is None else [
            OriginalMetric("final_loss", value, "raw", 3)
        ]
    if experiment == "e03":
        value = last_value(rows, "perplexity")
        return [] if value is None else [
            OriginalMetric("final_train_perplexity", value, "raw", 2)
        ]
    if experiment == "e04":
        metrics = []
        for split in ("train", "test"):
            value = last_value(rows, "perplexity", split=split)
            if value is not None:
                metrics.append(
                    OriginalMetric(f"final_{split}_perplexity", value, "raw", 2)
                )
        return metrics
    if experiment in {"e06", "e07"}:
        value = last_value(rows, "accuracy")
        return [] if value is None else [
            OriginalMetric("final_test_accuracy", value, "percent", 2, 100.0)
        ]
    raise ValueError(f"unsupported DS2 original summary: {experiment}")


def summarize(experiments: list[str], *, root: Path) -> list[Path]:
    return [
        write_experiment_summary(
            experiment=experiment,
            root=root,
            trial_ids=TRIAL_IDS[experiment],
            extractor=lambda rows, experiment=experiment: _metrics(experiment, rows),
        )
        for experiment in experiments
    ]
