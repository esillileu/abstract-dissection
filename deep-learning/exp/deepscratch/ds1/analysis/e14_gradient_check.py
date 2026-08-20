"""Compare ch05 numerical and backprop gradients and their runtimes."""

import csv

import matplotlib.pyplot as plt
import numpy as np

from exp.framework.analysis.core import mark_empty, save_figure

from .common import runs


PARAMETERS = ("W1", "b1", "W2", "b2")


def _rows(data, run):
    for path in ("observations/gradient_check.csv", "raw/gradient_check.csv"):
        rows = data.artifact_rows(run, path)
        if rows:
            return rows
    return []


def _timing(data, run):
    for path in ("observations/gradient_timing.csv", "raw/metrics.csv"):
        rows = data.artifact_rows(run, path)
        if rows:
            values = {}
            for row in rows:
                if "method" in row:
                    values[str(row["method"])] = float(row["seconds"])
                elif str(row.get("metric", "")).startswith("gradient_check/"):
                    values[str(row["metric"]).removeprefix("gradient_check/")] = float(row["value"])
            if values:
                return values
    return {}


def render(data, error_style, output):
    del error_style
    refs = runs(data, "GO03", ["TWO-LAYER-GRADIENT-CHECK"])["TWO-LAYER-GRADIENT-CHECK"]
    differences = {name: [] for name in PARAMETERS}
    timings = {name: [] for name in ("numerical", "backprop", "speedup")}
    for run in refs:
        for row in _rows(data, run):
            name = str(row["parameter"])
            if name in differences:
                differences[name].append(float(row["mean_absolute_difference"]))
        for name, value in _timing(data, run).items():
            normalized = name.removesuffix("_s")
            if normalized in timings:
                timings[normalized].append(value)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    if any(differences.values()):
        means = [np.mean(differences[name]) for name in PARAMETERS]
        stds = [np.std(differences[name], ddof=1) if len(differences[name]) > 1 else 0.0 for name in PARAMETERS]
        axes[0].bar(PARAMETERS, means, yerr=stds, capsize=3)
        axes[0].set_yscale("log")
    else:
        mark_empty(axes[0])
    axes[0].set(xlabel="parameter", ylabel="mean absolute difference")
    methods = ("numerical", "backprop")
    if all(timings[name] for name in methods):
        axes[1].bar(methods, [np.mean(timings[name]) for name in methods])
        axes[1].set_yscale("log")
    else:
        mark_empty(axes[1])
    axes[1].set(ylabel="gradient computation time (s)")
    save_figure(figure, output)
    plt.close(figure)

    summary = output.with_name(f"{output.stem}_summary.csv")
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("kind", "name", "mean", "standard_deviation", "run_count"))
        writer.writeheader()
        for kind, values in (("mean_absolute_difference", differences), ("timing", timings)):
            for name, samples in values.items():
                if samples:
                    writer.writerow({"kind": kind, "name": name, "mean": np.mean(samples), "standard_deviation": np.std(samples, ddof=1) if len(samples) > 1 else 0.0, "run_count": len(samples)})
    return [output, summary]
