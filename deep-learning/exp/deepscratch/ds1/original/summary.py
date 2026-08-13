"""Summaries of DS1 original-code result caches."""

from __future__ import annotations

from pathlib import Path

from exp.deepscratch.original_runtime.summary import (
    OriginalMetric,
    last_value,
    write_experiment_summary,
)


SUPPORTED_EXPERIMENTS = (
    "e01",
    "e02",
    "e03",
    "e04",
    "e05",
    "e06",
    "e07",
)
TRIAL_IDS = {
    "e01": tuple(
        f"dlfs1.ch06.optimizer-mnist.{name}"
        for name in ("sgd", "momentum", "adagrad", "adam")
    ),
    "e02": tuple(
        f"dlfs1.ch06.init-compare.{name}"
        for name in ("std-001", "xavier", "he")
    ),
    "e03": ("dlfs1.ch06.weight-decay.lambda-01",),
    "e04": ("dlfs1.ch06.dropout.on-ratio-02",),
    "e05": tuple(
        f"dlfs1.ch06.batchnorm.scale-{index:02d}.bn-{state}"
        for index in range(1, 17)
        for state in ("off", "on")
    ),
    "e06": ("dlfs1.ch07.simple-convnet",),
    "e07": ("dlfs1.ch08.deep-convnet",),
}


def _metrics(experiment: str, rows: list[dict[str, str]]) -> list[OriginalMetric]:
    if experiment in {"e01", "e02"}:
        value = last_value(rows, "value", metric="train/objective")
        return [] if value is None else [
            OriginalMetric("final_train_objective", value, "raw", 3)
        ]
    if experiment in {"e03", "e04"}:
        return _split_accuracies(rows, ("train", "test"))
    if experiment == "e05":
        value = last_value(rows, "accuracy")
        return [] if value is None else [
            OriginalMetric("final_train_accuracy", value, "percent", 2, 100.0)
        ]
    if experiment in {"e06", "e07"}:
        return _split_accuracies(rows, ("train", "test", "test-full"))
    raise ValueError(f"unsupported DS1 original summary: {experiment}")


def _split_accuracies(
    rows: list[dict[str, str]],
    splits: tuple[str, ...],
) -> list[OriginalMetric]:
    metrics = []
    for split in splits:
        value = last_value(rows, "accuracy", split=split)
        if value is not None:
            metrics.append(
                OriginalMetric(
                    f"final_{split.replace('-', '_')}_accuracy",
                    value,
                    "percent",
                    2,
                    100.0,
                )
            )
    return metrics


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
