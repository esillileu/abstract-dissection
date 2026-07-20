"""Reusable mean/min-max curve renderer for deepbase2 experiments."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .common import curve, latest_seeded_runs, plot_band, write_summary


def render(*, mlflow_client, experiment_name: str, atomic_run_ids: list[str], metric: str, title: str, ylabel: str, output, summary_csv, marker: str | None = None) -> None:
    grouped = latest_seeded_runs(mlflow_client, experiment_name, atomic_run_ids)
    curves = {atomic: curve(mlflow_client, runs, metric) for atomic, runs in grouped.items()}
    figure, axis = plt.subplots(figsize=(9, 5))
    for atomic in atomic_run_ids:
        plot_band(axis, curves[atomic], label=atomic, marker=marker)
    axis.set_title(title)
    axis.set_xlabel("step" if metric.startswith("step/") else "epoch")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    write_summary(summary_csv, grouped, curves, metric)
