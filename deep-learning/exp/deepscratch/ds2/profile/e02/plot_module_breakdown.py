"""Render cumulative implemented Word2Vec module timings from JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from exp.framework.analysis.core import save_figure
from exp.framework.plotting.theme import ACCENT_COLORS


CONDITION_ORDER = (
    "implemented-cbow-fs",
    "implemented-skipgram-fs",
    "implemented-cbow-ns",
    "implemented-skipgram-ns",
    "implemented-cbow-fused-ns",
    "implemented-skipgram-fused-ns",
)

COMPONENT_ORDER = {
    "separate_model_objective": (
        "batch_adapter",
        "objective_prepare",
        "model_forward",
        "objective_forward",
        "objective_backward",
        "model_backward",
        "optimizer",
    ),
    "fused_negative_sampling": (
        "batch_adapter",
        "objective_prepare",
        "fused_forward_loss",
        "fused_backward",
        "optimizer",
    ),
}

LABELS = {
    "implemented-cbow-fs": "CBOW\nFS",
    "implemented-skipgram-fs": "Skip-gram\nFS",
    "implemented-cbow-ns": "CBOW\nNS",
    "implemented-skipgram-ns": "Skip-gram\nNS",
    "implemented-cbow-fused-ns": "CBOW\nFused NS",
    "implemented-skipgram-fused-ns": "Skip-gram\nFused NS",
}


def render(input_path: Path, output_path: Path, csv_path: Path) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload["results"]
    timings = {
        condition: {
            row["component"]: float(row["timing"]["mean_ms"])
            for row in rows
            if row["condition"] == condition
        }
        for condition in CONDITION_ORDER
    }

    figure, axis = plt.subplots(figsize=(12, 7))
    bottoms = [0.0] * len(CONDITION_ORDER)
    csv_rows = []
    for index, scope in enumerate(
        ("separate_model_objective", "fused_negative_sampling")
    ):
        components = COMPONENT_ORDER[scope]
        for component_index, component in enumerate(components):
            values = [
                timings[condition].get(component, 0.0)
                if _scope(condition, rows) == scope
                else 0.0
                for condition in CONDITION_ORDER
            ]
            axis.bar(
                range(len(CONDITION_ORDER)),
                values,
                bottom=bottoms,
                label=component.replace("_", " ").title(),
                color=ACCENT_COLORS[
                    (component_index + index * 3) % len(ACCENT_COLORS)
                ],
            )
            bottoms = [left + right for left, right in zip(bottoms, values)]

    for condition, total in zip(CONDITION_ORDER, bottoms):
        csv_rows.append((condition, total))
        axis.text(
            CONDITION_ORDER.index(condition),
            total,
            f"{total:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axis.set_xticks(range(len(CONDITION_ORDER)), [LABELS[c] for c in CONDITION_ORDER])
    axis.set_ylabel("Mean component time (ms/update)")
    axis.set_title("DS2 Word2Vec cumulative module time\n(warmup 50, measured 100)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3, fontsize=9)
    figure.tight_layout()
    save_figure(figure, output_path)
    plt.close(figure)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "condition,total_mean_ms_per_update\n"
        + "\n".join(f"{condition},{total:.9f}" for condition, total in csv_rows)
        + "\n",
        encoding="utf-8",
    )


def _scope(condition: str, rows: list[dict[str, object]]) -> str:
    for row in rows:
        if row["condition"] == condition:
            return str(row["measurement_scope"])
    raise KeyError(condition)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    render(args.input, args.output, args.csv)


if __name__ == "__main__":
    main()
