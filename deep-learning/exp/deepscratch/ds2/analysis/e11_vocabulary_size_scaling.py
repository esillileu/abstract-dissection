"""DS2 E11: canonical Word2Vec vocabulary-size scaling analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from exp.framework.analysis.core import save_figure
from exp.framework.plotting.theme import ACCENT_COLORS, MUTED


ATOMIC_IDS = (
    "PF-VSCALE-CBOW-FS",
    "PF-VSCALE-CBOW-NS",
    "PF-VSCALE-CBOW-FUSED-NS",
    "PF-VSCALE-SKIPGRAM-FS",
    "PF-VSCALE-SKIPGRAM-NS",
    "PF-VSCALE-SKIPGRAM-FUSED-NS",
)
STYLES = {
    "full_softmax": ("Full Softmax", ACCENT_COLORS[0], "o", "-"),
    "negative_sampling": ("Negative Sampling", ACCENT_COLORS[1], "s", "--"),
    "fused_negative_sampling": (
        "Fused Negative Sampling", ACCENT_COLORS[3], "^", "-."
    ),
}


def render(
    data,
    error_style: str,
    output: Path,
    *,
    title_fontsize: float | None = 18,
    legend_fontsize: float | None = 11,
) -> list[Path]:
    del error_style
    rows = _scaling_rows(data)
    outputs = []
    for model in ("CBOW", "Skip-gram"):
        figure, axis = plt.subplots(figsize=(6.4, 4.8))
        _plot_model(axis, rows, model, legend_fontsize=legend_fontsize)
        path = output.with_name(f"{output.stem}_{_model_slug(model)}{output.suffix}")
        save_figure(figure, path)
        plt.close(figure)
        outputs.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8), squeeze=False)
    for axis, model in zip(axes[0], ("CBOW", "Skip-gram"), strict=True):
        _plot_model(
            axis,
            rows,
            model,
            title=model,
            title_fontsize=title_fontsize,
            legend_fontsize=legend_fontsize,
        )
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)
    outputs.insert(0, output)

    csv_path = output.with_name(f"{output.stem}_scaling.csv")
    _write_rows(csv_path, rows)
    outputs.append(csv_path)
    return outputs


def _scaling_rows(data) -> list[dict[str, object]]:
    rows = []
    grouped = data.runs(ATOMIC_IDS)
    for atomic_id in ATOMIC_IDS:
        for run in grouped[atomic_id]:
            artifact = data.artifact_file(run, "profile/result.json")
            if artifact is None:
                continue
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            if payload.get("schema_name") != "ds2-profile":
                raise ValueError(f"run {run.run_id} has an incompatible profile schema")
            for point in payload.get("points", []):
                if point.get("status") != "ok":
                    continue
                metrics = point.get("metrics", {})
                rows.append({
                    "condition": atomic_id,
                    "model": _condition_model(atomic_id),
                    "implementation": _condition_implementation(atomic_id),
                    "vocabulary_size": int(point["axes"]["vocabulary_size"]),
                    "mean_ms": float(metrics["update_ms"]),
                    "ci95_lower_ms": metrics.get("ci95_lower_ms"),
                    "ci95_upper_ms": metrics.get("ci95_upper_ms"),
                    "run_id": run.run_id,
                })
    return rows


def _plot_model(
    axis,
    rows: list[dict[str, object]],
    model: str,
    *,
    title: str | None = None,
    title_fontsize: float | None = None,
    legend_fontsize: float | None = None,
) -> None:
    selected = [row for row in rows if row["model"] == model]
    for implementation, (label, color, marker, linestyle) in STYLES.items():
        points = sorted(
            (row for row in selected if row["implementation"] == implementation),
            key=lambda row: int(row["vocabulary_size"]),
        )
        if not points:
            continue
        axis.plot(
            [int(row["vocabulary_size"]) for row in points],
            [float(row["mean_ms"]) for row in points],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
        )
    axis.set(
        xlabel="Vocabulary size",
        ylabel="Mean update time (ms)",
        xscale="log",
        yscale="linear",
    )
    if title is not None:
        axis.set_title(title, fontsize=title_fontsize)
    axis.grid(True, which="both", alpha=0.25, color=MUTED)
    if selected:
        axis.legend(fontsize=legend_fontsize)


def _condition_model(condition: str) -> str:
    return "CBOW" if "-CBOW-" in condition else "Skip-gram"


def _condition_implementation(condition: str) -> str:
    if condition.endswith("-FUSED-NS"):
        return "fused_negative_sampling"
    if condition.endswith("-NS"):
        return "negative_sampling"
    return "full_softmax"


def _model_slug(model: str) -> str:
    return "cbow" if model == "CBOW" else "skipgram"


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "condition",
        "model",
        "implementation",
        "vocabulary_size",
        "mean_ms",
        "ci95_lower_ms",
        "ci95_upper_ms",
        "run_id",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["render"]
