"""DS2 E10: canonical Word2Vec operation-time breakdown analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from exp.framework.analysis.core import save_figure
from exp.framework.plotting.theme import ACCENT_COLORS


ATOMIC_IDS = (
    "PF-W2V-CBOW-IMPLEMENTED-FS",
    "PF-W2V-CBOW-IMPLEMENTED-NS",
    "PF-W2V-CBOW-IMPLEMENTED-FUSED-NS",
    "PF-W2V-SKIPGRAM-IMPLEMENTED-FS",
    "PF-W2V-SKIPGRAM-IMPLEMENTED-NS",
    "PF-W2V-SKIPGRAM-IMPLEMENTED-FUSED-NS",
)


def render(
    data,
    error_style: str,
    output: Path,
    *,
    title_fontsize: float | None = 22,
    legend_fontsize: float | None = 9,
) -> list[Path]:
    del error_style
    rows = _module_rows(data)
    outputs = []
    for model in ("CBOW", "Skip-gram"):
        figure, axis = plt.subplots(figsize=(7, 6))
        _plot_model(axis, rows, model, legend_fontsize=legend_fontsize)
        path = output.with_name(f"{output.stem}_{_model_slug(model)}{output.suffix}")
        save_figure(figure, path)
        plt.close(figure)
        outputs.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), squeeze=False)
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

    csv_path = output.with_name(f"{output.stem}_operations.csv")
    _write_rows(csv_path, rows)
    outputs.append(csv_path)
    return outputs


def _module_rows(data) -> list[dict[str, object]]:
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
                for module in point.get("sections", {}).get("modules", []):
                    timing = module.get("timing", {})
                    rows.append({
                        "condition": atomic_id,
                        "component": str(module.get("component", "")),
                        "mean_ms": float(timing.get("mean_ms", 0.0)),
                        "stdev_ms": float(timing.get("stdev_ms", 0.0)),
                        "measurement_scope": str(
                            module.get("measurement_scope", "")
                        ),
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
    selected = [row for row in rows if _condition_model(row["condition"]) == model]
    conditions = [
        condition
        for condition in ATOMIC_IDS
        if _condition_model(condition) == model
        and any(row["condition"] == condition for row in selected)
    ]
    components = sorted({str(row["component"]) for row in selected})
    bottoms = [0.0] * len(conditions)
    for index, component in enumerate(components):
        values = [
            next(
                (
                    float(row["mean_ms"])
                    for row in selected
                    if row["condition"] == condition
                    and row["component"] == component
                ),
                0.0,
            )
            for condition in conditions
        ]
        axis.bar(
            range(len(conditions)),
            values,
            bottom=bottoms,
            label=component.replace("_", " ").title(),
            color=ACCENT_COLORS[index % len(ACCENT_COLORS)],
        )
        bottoms = [left + right for left, right in zip(bottoms, values)]
    axis.set_xticks(
        range(len(conditions)),
        [_condition_label(condition) for condition in conditions],
        rotation=0,
        ha="center",
    )
    axis.tick_params(axis="x", labelsize=12)
    for label in axis.get_xticklabels():
        label.set_fontweight("bold")
    axis.set_ylabel("Operation time (ms)")
    if title is not None:
        axis.set_title(title, fontsize=title_fontsize)
    if components:
        axis.legend(fontsize=legend_fontsize, ncol=2)
    axis.grid(axis="y", alpha=0.25)


def _condition_model(condition: object) -> str:
    return "CBOW" if "-CBOW-" in str(condition) else "Skip-gram"


def _condition_label(condition: str) -> str:
    tokens = [
        token
        for token in condition.removeprefix("PF-W2V-").split("-")[1:]
        if token != "IMPLEMENTED"
    ]
    names = {
        "ORIGINAL": "Original",
        "IMPLEMENTED": "Implemented",
        "ONEHOT": "One-Hot Input",
        "FS": "Full Softmax",
        "NS": "Negative Sampling",
        "FUSED": "Fused",
    }
    label = " ".join(names.get(token, token.title()) for token in tokens)
    return label.replace("Negative Sampling", "Negative\nSampling").replace(
        "Fused Negative Sampling", "Fused Negative\nSampling"
    )


def _model_slug(model: str) -> str:
    return "cbow" if model == "CBOW" else "skipgram"


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "condition",
        "component",
        "mean_ms",
        "stdev_ms",
        "measurement_scope",
        "run_id",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["render"]
