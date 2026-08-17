"""Render canonical e11/PF02 vocabulary-size scaling runs from MLflow."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

from exp.deepscratch.profile.selection import latest_profile_runs
from exp.deepscratch.ds2.profile.word2vec.scaling import (
    render_scaling,
    summarize_crossovers,
)


def render(
    tracking_uri: str,
    output_dir: Path,
    *,
    device: str | None = None,
    timing_source: str | None = None,
) -> tuple[Path, Path, Path]:
    client = MlflowClient(tracking_uri=tracking_uri)
    rows = []
    for run in latest_profile_runs(
        client, experiment_name="deepscratch.ds2", study_id="e11",
        device=device, timing_source=timing_source,
        schema_name="ds2-profile",
        protocol_version="ds2-e11-vocabulary-size-scaling-v1",
    ):
        condition = _condition_from_atomic(run.data.tags["atomic_run.id"])
        model = "CBOW" if "-cbow-" in condition else "SkipGram"
        objective = (
            "FusedNegativeSampling"
            if condition.endswith("-fused-ns")
            else "NegativeSampling"
            if condition.endswith("-ns")
            else "FullSoftmax"
        )
        points = {}
        for key, value in run.data.metrics.items():
            prefix, suffix = "profile/vocabulary_size/", "/update_ms"
            if not key.startswith(prefix) or not key.endswith(suffix):
                continue
            vocabulary_size = int(key[len(prefix) : -len(suffix)])
            points[vocabulary_size] = float(value)
        for vocabulary_size, value in points.items():
            metric_prefix = f"profile/vocabulary_size/{vocabulary_size}"
            rows.append({
                "condition": condition,
                "model": model,
                "objective": objective,
                "device": run.data.params.get(
                    "numerics/device", run.data.tags.get("runtime.device_type", "")
                ),
                "timing_source": run.data.params.get(
                    "profiling/timing_source", ""
                ),
                "vocab_size": vocabulary_size,
                "update_ms": value,
                "status": "ok",
                "ci95_lower_ms": run.data.metrics.get(
                    f"{metric_prefix}/ci95_lower_ms"
                ),
                "ci95_upper_ms": run.data.metrics.get(
                    f"{metric_prefix}/ci95_upper_ms"
                ),
                "run_id": run.info.run_id,
            })
    if not rows:
        raise ValueError("no durable FINISHED e11/PF02 scaling runs were found")
    rows.sort(key=lambda row: (str(row["condition"]), int(row["vocab_size"])))
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if device is None else f"_{device.replace(':', '')}"
    csv_path = output_dir / f"ds2_e11_vocabulary_size_scaling{suffix}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure = render_scaling({
        "metadata": {"device": "recorded", "timing_source": "window"},
        "results": rows,
    })
    png_path = output_dir / f"ds2_e11_vocabulary_size_scaling{suffix}.png"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    markdown_path = output_dir / f"ds2_e11_vocabulary_size_scaling{suffix}.md"
    lines = [
        "# DS2 e11 / PF02 vocabulary-size scaling",
        "",
        "Source study: `e02`. The independent variable is vocabulary size `V`.",
        "",
        "| condition | V | update (ms) |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {row['condition']} | {int(row['vocab_size'])} | "
        f"{float(row['update_ms']):.3f} |"
        for row in rows
    )
    crossovers = summarize_crossovers(rows)
    if crossovers:
        lines.extend(["", "## Crossovers", ""])
        for model, result in crossovers.items():
            first = result.get(
                "first_confirmed_negative_sampling_win_vocab_size"
            )
            observed = result.get(
                "first_observed_negative_sampling_win_vocab_size"
            )
            lines.append(
                f"- {model}: first point-estimate win V="
                f"{observed if observed is not None else 'not observed'}; "
                f"first confirmed win V="
                f"{first if first is not None else 'not observed'}"
            )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return png_path, csv_path, markdown_path


def _condition_from_atomic(atomic_run_id: str) -> str:
    model = "cbow" if "-CBOW-" in atomic_run_id else "skipgram"
    objective = (
        "fused-ns"
        if atomic_run_id.endswith("-FUSED-NS")
        else "ns"
        if atomic_run_id.endswith("-NS")
        else "fs"
    )
    return f"implemented-{model}-{objective}"


__all__ = ["render"]
