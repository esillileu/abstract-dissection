"""Seed-aware MLflow analysis for promoted original-source runs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exp.analyze import mlflow_client, tracking_uri_default


def analyze(*, domain: str, experiments: list[str], tracking_uri: str | None, error_style: str, output_dir: Path | None, seed: int | None, summary: bool) -> None:
    selected = _experiments(domain, experiments)
    client = mlflow_client(tracking_uri or tracking_uri_default())
    experiment = client.get_experiment_by_name(domain)
    if experiment is None:
        raise ValueError(f"MLflow experiment does not exist: {domain}")
    root = output_dir or Path("exp") / domain / "results" / "image"
    root.mkdir(parents=True, exist_ok=True)
    for experiment_id in selected:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"attributes.status = 'FINISHED' and tags.`run.type` = 'seed_trial' and tags.`experiment.ids` = '{experiment_id}'",
            max_results=5000,
        )
        if seed is not None:
            runs = [run for run in runs if str(run.data.params.get("seed/master", run.data.tags.get("master_seed", ""))) == str(seed)]
        if not runs:
            raise ValueError(f"no completed runs for {domain}/{experiment_id}")
        suffix = "" if seed is None else f"_seed-{seed}"
        if summary:
            output = root / f"{experiment_id}_summary{suffix}.csv"
            _summary(output, runs)
        else:
            output = root / f"{experiment_id}_{error_style}{suffix}.png"
            _curves(client, output, runs, error_style)
            if domain == "ds1_original" and experiment_id == "e06":
                _filters(client, output.with_name(f"{experiment_id}_filters{suffix}.png"), runs)
        print(output)


def _experiments(domain: str, requested: list[str]) -> list[str]:
    if not requested:
        return [path.name[:3] for path in sorted((Path("exp") / domain / "config").glob("e[0-9][0-9]_*.yaml"))]
    from exp.parsing import parse_experiment_ids

    return parse_experiment_ids(requested)


def _summary(path: Path, runs) -> None:
    keys = sorted({key for run in runs for key in run.data.metrics if key.startswith("final/") or key.startswith("runtime/")})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["atomic_run_id", "metric", "count", "mean", "std"])
        grouped = defaultdict(list)
        for run in runs:
            atomic = run.data.tags.get("atomic_run.id", "")
            for key in keys:
                if key in run.data.metrics:
                    grouped[(atomic, key)].append(float(run.data.metrics[key]))
        for (atomic, key), values in sorted(grouped.items()):
            writer.writerow([atomic, key, len(values), float(np.mean(values)), float(np.std(values))])


def _curves(client, path: Path, runs, error_style: str) -> None:
    grouped = defaultdict(list)
    for run in runs:
        atomic = run.data.tags.get("atomic_run.id", "")
        metric_keys = sorted(
            (
                key
                for key in run.data.metrics
                if key.startswith(("valid/", "train/", "test/", "observation/"))
            ),
            key=lambda key: (
                next(
                    index
                    for index, prefix in enumerate(
                        ("valid/", "train/", "test/", "observation/")
                    )
                    if key.startswith(prefix)
                ),
                key,
            ),
        )
        if not metric_keys:
            continue
        key = metric_keys[0]
        history = client.get_metric_history(run.info.run_id, key)
        if history:
            grouped[atomic].append([(item.step, item.value) for item in history])
    if not grouped:
        raise ValueError("selected runs expose no curve metric history")
    figure, axis = plt.subplots(figsize=(8, 5))
    summary_rows = []
    for atomic, histories in sorted(grouped.items()):
        common = sorted(set.intersection(*(set(step for step, _ in history) for history in histories)))
        values = np.asarray([[dict(history)[step] for step in common] for history in histories], dtype=float)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        axis.plot(common, mean, label=atomic)
        if len(histories) > 1:
            if error_style == "band":
                axis.fill_between(common, mean - std, mean + std, alpha=0.2)
            else:
                axis.errorbar(common, mean, yerr=std, fmt="none", alpha=0.4)
        summary_rows.extend((atomic, step, avg, spread, len(histories)) for step, avg, spread in zip(common, mean, std, strict=True))
    axis.set_xlabel("step")
    axis.set_ylabel("metric")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["atomic_run_id", "step", "mean", "std", "seed_count"])
        writer.writerows(summary_rows)


def _filters(client, path: Path, runs) -> None:
    selected = min(
        runs,
        key=lambda run: int(run.data.params.get("seed/master", run.data.tags.get("master_seed", 0))),
    )
    try:
        checkpoint = Path(client.download_artifacts(selected.info.run_id, "raw/checkpoint.npz"))
    except Exception as exc:
        raise ValueError("DS1 e06 filter artifact is missing: raw/checkpoint.npz") from exc
    with np.load(checkpoint, allow_pickle=False) as archive:
        initial = archive["initial_W1"]
        final_key = "param__W1" if "param__W1" in archive.files else "W1"
        final = archive[final_key]
    count = min(16, len(initial))
    figure, axes = plt.subplots(4, 8, figsize=(10, 5))
    for index in range(16):
        for column, values in enumerate((initial, final)):
            axis = axes[index // 4, (index % 4) * 2 + column]
            axis.axis("off")
            if index < count:
                axis.imshow(values[index, 0], cmap="gray")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
