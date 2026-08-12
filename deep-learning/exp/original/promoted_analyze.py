"""Seed-aware MLflow analysis for promoted original-source runs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exp.analyze import aggregate, mark_empty, mlflow_client, plot_curve, save_figure, tracking_uri_default, write_summary


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
            order_by=["attributes.start_time DESC"],
            max_results=5000,
        )
        runs = _latest_seed_runs(runs, seed=seed)
        if not runs:
            raise ValueError(f"no completed runs for {domain}/{experiment_id}")
        suffix = "" if seed is None else f"_seed-{seed}"
        if summary:
            output = root / f"{experiment_id}_summary{suffix}.csv"
            _summary(
                output,
                runs,
                domain=domain,
                experiment_id=experiment_id,
                client=client,
            )
        else:
            output = root / f"{experiment_id}_{error_style}{suffix}.png"
            _curves(
                client,
                output,
                runs,
                error_style,
                labels=_curve_labels(domain, experiment_id),
                domain=domain,
                experiment_id=experiment_id,
            )
            if domain == "ds1_original" and experiment_id == "e06":
                _filters(client, output.with_name(f"{experiment_id}_filters{suffix}.png"), runs)
        print(output)


def _latest_seed_runs(runs, *, seed: int | None = None):
    """Keep the newest completed attempt for every atomic ID and seed."""

    selected = {}
    for run in sorted(
        runs,
        key=lambda item: int(item.info.start_time or 0),
        reverse=True,
    ):
        atomic_run_id = run.data.tags.get("atomic_run.id", "")
        master_seed = str(
            run.data.params.get(
                "seed/master",
                run.data.tags.get("master_seed", run.info.run_id),
            )
        )
        if seed is not None and master_seed != str(seed):
            continue
        selected.setdefault((atomic_run_id, master_seed), run)
    return list(selected.values())


def _experiments(domain: str, requested: list[str]) -> list[str]:
    if not requested:
        return [path.name[:3] for path in sorted((Path("exp") / domain / "config").glob("e[0-9][0-9]_*.yaml"))]
    from exp.parsing import parse_experiment_ids

    return parse_experiment_ids(requested)


def _summary(
    path: Path,
    runs,
    *,
    domain: str,
    experiment_id: str,
    client=None,
) -> None:
    raw_metrics = {
        run.info.run_id: _ds1_cnn_final_metrics(client, run, domain)
        for run in runs
    } if domain == "ds1_original" and experiment_id in {"e06", "e07"} else {}
    keys = sorted(
        {
            key
            for run in runs
            for key in (
                set(run.data.metrics).union(
                    raw_metrics.get(
                        getattr(getattr(run, "info", None), "run_id", ""),
                        {},
                    )
                )
            )
            if key.startswith("final/") or key.startswith("runtime/")
        }
    )
    specs = _summary_specs(domain, experiment_id, keys)
    grouped = defaultdict(list)
    for run in runs:
        atomic = run.data.tags.get("atomic_run.id", "")
        run_metrics = {
            **run.data.metrics,
            **raw_metrics.get(
                getattr(getattr(run, "info", None), "run_id", ""),
                {},
            ),
        }
        for key, _label, _scale, _decimals, _unit in specs:
            if key in run_metrics:
                grouped[(atomic, key)].append(float(run_metrics[key]))
    rows = []
    for (atomic, key), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        rows.append(
            (
                atomic,
                key,
                len(values),
                float(array.mean()),
                float(array.std(ddof=1)) if len(values) > 1 else 0.0,
                float(array.min()),
                float(array.max()),
            )
        )
    print(f"{domain}/{experiment_id} summary (mean ± sample standard deviation; min-max)")
    current_atomic = None
    for atomic, key, count, mean, standard_deviation, minimum, maximum in rows:
        if atomic != current_atomic:
            print(f"[{atomic}]")
            current_atomic = atomic
        label, scale, decimals, unit = next(
            (label, scale, decimals, unit)
            for spec_key, label, scale, decimals, unit in specs
            if spec_key == key
        )
        print(
            f"{label}{unit}: {mean * scale:.{decimals}f} ± "
            f"{standard_deviation * scale:.{decimals}f}, "
            f"[{minimum * scale:.{decimals}f}, {maximum * scale:.{decimals}f}], "
            f"n={count}"
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "atomic_run_id",
                "metric",
                "count",
                "mean",
                "standard_deviation",
                "minimum",
                "maximum",
            ]
        )
        writer.writerows(rows)


def _summary_specs(domain: str, experiment_id: str, keys: list[str]):
    """Map projected MLflow metrics back to the original summary vocabulary."""

    if domain == "ds1_original":
        if experiment_id in {"e01", "e02"}:
            wanted = (("final/train/loss", "final_train_objective", 1.0, 3, ""),)
        elif experiment_id in {"e03", "e04"}:
            wanted = tuple(
                (f"final/{split}/accuracy", f"final_{split}_accuracy", 100.0, 2, " (%)")
                for split in ("train", "test")
            )
        elif experiment_id == "e05":
            wanted = (("final/test/accuracy", "final_train_accuracy", 100.0, 2, " (%)"),)
        elif experiment_id in {"e06", "e07"}:
            wanted = tuple(
                (f"final/{split}/accuracy", f"final_{split.replace('-', '_')}_accuracy", 100.0, 2, " (%)")
                for split in ("train", "test", "test-full")
            )
        else:
            wanted = ()
    elif domain == "ds2_original":
        if experiment_id in {"e01", "e02"}:
            wanted = (("final/train/loss", "final_loss", 1.0, 3, ""),)
        elif experiment_id == "e03":
            wanted = (("final/train/perplexity", "final_train_perplexity", 1.0, 2, ""),)
        elif experiment_id == "e04":
            wanted = tuple(
                (f"final/{split}/perplexity", f"final_{split}_perplexity", 1.0, 2, "")
                for split in ("train", "test")
            )
        elif experiment_id in {"e06", "e07"}:
            wanted = (("final/test/accuracy", "final_test_accuracy", 100.0, 2, " (%)"),)
        else:
            wanted = ()
    else:
        wanted = ()
    return tuple(spec for spec in wanted if spec[0] in keys or spec[0] == "runtime/train_total_s") + (
        ("runtime/train_total_s", "training_time", 1.0, 1, " (s)"),
    )


def _curve_labels(domain: str, experiment_id: str) -> dict[str, str] | None:
    """Return legacy visualization labels for analyses with a fixed vocabulary."""

    if domain == "ds2_original" and experiment_id == "e06":
        return {
            "SEQ2SEQ-FORWARD": "vanilla / forward",
            "SEQ2SEQ-REVERSE": "vanilla / reverse",
            "PEEKY-FORWARD": "peeky / forward",
            "PEEKY-REVERSE": "peeky / reverse",
        }
    return None


def _curves(
    client,
    path: Path,
    runs,
    error_style: str,
    *,
    labels=None,
    domain: str | None = None,
    experiment_id: str | None = None,
) -> None:
    if domain == "ds1_original" and experiment_id in {"e06", "e07"}:
        return _ds1_cnn_curves(client, path, runs, error_style)
    grouped = defaultdict(list)
    metric_names = set()
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
        metric_names.add(key)
        history = {
            float(item.step): float(item.value)
            for item in client.get_metric_history(run.info.run_id, key)
            if np.isfinite(item.value)
        }
        if history:
            grouped[atomic].append(history)
    if not grouped:
        raise ValueError("selected runs expose no curve metric history")
    figure, axis = plt.subplots()
    figure._analysis_match_original_canvas = True
    order = {atomic: index for index, atomic in enumerate(labels or {})}
    items = sorted(grouped.items(), key=lambda item: (order.get(item[0], len(order)), item[0]))
    for atomic, histories in items:
        curve = aggregate(histories)
        plot_curve(
            axis,
            curve,
            label=(labels or {}).get(atomic, atomic),
            marker="o",
            error_style=error_style,
            error_every=5,
        )
    if any("/loss" in key or key.endswith("loss") for key in metric_names):
        axis.set(xlabel="iterations", ylabel="loss", ylim=(0, 1))
    elif any("perplexity" in key for key in metric_names):
        axis.set(xlabel="iterations (x20)", ylabel="perplexity")
    else:
        axis.set(xlabel="epochs", ylabel="accuracy", ylim=(0, 1))
    mark_empty(axis)
    if axis.has_data():
        axis.legend()
    save_figure(figure, path)
    plt.close(figure)
    write_summary(path.with_suffix(".csv"), {
        atomic: aggregate(histories)
        for atomic, histories in items
    })


def _ds1_cnn_curves(client, path: Path, runs, error_style: str) -> None:
    grouped = defaultdict(list)
    for run in runs:
        atomic = run.data.tags.get("atomic_run.id", "")
        rows = _raw_metric_rows(client, run, "ds1_original")
        for split in ("train", "test"):
            history = {}
            for row in rows:
                if row.get("split") != split:
                    continue
                try:
                    epoch = float(row["epoch"])
                    accuracy = float(row["accuracy"])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(epoch) and np.isfinite(accuracy):
                    history[epoch] = accuracy
            if history:
                grouped[f"{atomic}/{split}"].append(history)
    if not grouped:
        raise ValueError("selected DS1 original runs expose no raw accuracy curves")
    figure, axis = plt.subplots()
    figure._analysis_match_original_canvas = True
    items = sorted(
        grouped.items(),
        key=lambda item: (item[0].rsplit("/", 1)[0], item[0].endswith("/test")),
    )
    for series, histories in items:
        plot_curve(
            axis,
            aggregate(histories),
            label=series,
            marker="o" if series.endswith("/train") else "s",
            error_style=error_style,
            error_every=2,
        )
    axis.set(xlabel="epochs", ylabel="accuracy", ylim=(0, 1))
    axis.legend(loc="lower right")
    save_figure(figure, path)
    plt.close(figure)
    write_summary(
        path.with_suffix(".csv"),
        {series: aggregate(histories) for series, histories in items},
    )


def _ds1_cnn_final_metrics(client, run, domain: str) -> dict[str, float]:
    values = {}
    positions = {}
    for position, row in enumerate(_raw_metric_rows(client, run, domain)):
        split = row.get("split", "")
        if split not in {"train", "test", "test-full"}:
            continue
        try:
            epoch = float(row["epoch"])
            accuracy = float(row["accuracy"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(epoch) or not np.isfinite(accuracy):
            continue
        key = f"final/{split}/accuracy"
        if (epoch, position) >= positions.get(key, (-np.inf, -1)):
            positions[key] = (epoch, position)
            values[key] = accuracy
    return values


def _raw_metric_rows(client, run, domain: str) -> list[dict[str, str]]:
    path = _local_raw_metric_path(run, domain)
    if path is None and client is not None:
        try:
            path = Path(client.download_artifacts(run.info.run_id, "raw/metrics.csv"))
        except Exception:
            return []
    if path is None or not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except OSError:
        return []


def _local_raw_metric_path(run, domain: str) -> Path | None:
    run_key = run.data.tags.get("run.key")
    if not run_key:
        return None
    path = Path("exp") / domain / "results/mlflow_artifacts" / run_key / "raw/metrics.csv"
    return path if path.is_file() else None


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
