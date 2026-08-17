"""Render the canonical e10/PF01 MLflow profile runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

from exp.deepscratch.profile.selection import latest_profile_runs
from exp.framework.plotting.theme import ACCENT_COLORS, apply_plot_theme


METRICS = (
    "profile/update/mean_ms",
    "profile/update/stdev_ms",
    "profile/update/cold_ms",
    "profile/epoch/estimated_seconds",
    "profile/total/estimated_seconds",
)


def render(
    tracking_uri: str,
    output_dir: Path,
    *,
    device: str | None = None,
    timing_source: str | None = None,
) -> tuple[Path, ...]:
    client = MlflowClient(tracking_uri=tracking_uri)
    rows = []
    selected_runs = latest_profile_runs(
        client, experiment_name="deepscratch.ds2", study_id="e10",
        device=device, timing_source=timing_source,
        schema_name="ds2-profile", protocol_version="ds2-e10-profile-v1",
    )
    for run in selected_runs:
        tags = run.data.tags
        params = run.data.params
        condition = tags["atomic_run.id"]
        rows.append({
            "condition": condition,
            "subject_variant": tags.get("profile.subject_variant", ""),
            "device": params.get("numerics/device", tags.get("runtime.device_type", "")),
            "timing_source": params.get("profiling/timing_source", ""),
            "run_id": run.info.run_id,
            **{metric: run.data.metrics.get(metric, "") for metric in METRICS},
        })
    if not rows:
        raise ValueError("no durable FINISHED e10/PF01 profile runs were found")
    rows.sort(key=lambda row: str(row["condition"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if device is None else f"_{device.replace(':', '')}"
    csv_path = output_dir / f"ds2_e10_profile_results{suffix}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    apply_plot_theme()
    figure, axis = plt.subplots(figsize=(12, 6))
    labels = [str(row["condition"]).removeprefix("PF-W2V-") for row in rows]
    means = [float(row["profile/update/mean_ms"]) for row in rows]
    errors = [float(row["profile/update/stdev_ms"]) for row in rows]
    colors = [
        ACCENT_COLORS[0] if row["subject_variant"] == "original" else ACCENT_COLORS[1]
        for row in rows
    ]
    axis.bar(range(len(rows)), means, yerr=errors, color=colors, capsize=3)
    axis.set_xticks(range(len(rows)), labels, rotation=55, ha="right")
    axis.set_ylabel("Update time (ms)")
    axis.set_title("DS2 e10 / PF01 — PTB Word2Vec profile")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    png_path = output_dir / f"ds2_e10_profile_update_comparison{suffix}.png"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    markdown_path = output_dir / f"ds2_e10_profile_report{suffix}.md"
    lines = [
        "# DS2 e10 / PF01 profile",
        "",
        "Source study: `e02`.",
        "",
        "| condition | subject | device | update mean ± stdev (ms) |",
        "|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['condition']} | {row['subject_variant']} | {row['device']} | "
            f"{float(row['profile/update/mean_ms']):.3f} ± "
            f"{float(row['profile/update/stdev_ms']):.3f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    module_rows = _load_module_rows(client, selected_runs)
    module_csv = output_dir / f"ds2_e10_profile_module_breakdown{suffix}.csv"
    module_fields = (
        "condition", "component", "mean_ms", "stdev_ms",
        "measurement_scope", "run_id",
    )
    with module_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=module_fields)
        writer.writeheader()
        writer.writerows(module_rows)
    module_png = output_dir / f"ds2_e10_profile_module_breakdown{suffix}.png"
    _render_modules(module_rows, module_png)
    return png_path, csv_path, markdown_path, module_png, module_csv


def _load_module_rows(client: MlflowClient, runs) -> list[dict[str, object]]:
    rows = []
    for run in runs:
        path = Path(client.download_artifacts(run.info.run_id, "profile/result.json"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_name") != "ds2-profile":
            raise ValueError(
                f"run {run.info.run_id} has an incompatible profile result schema"
            )
        for point in payload.get("points", []):
            modules = point.get("sections", {}).get("modules", [])
            for module in modules:
                timing = module.get("timing", {})
                rows.append({
                    "condition": point.get("condition_id", ""),
                    "component": module.get("component", ""),
                    "mean_ms": float(timing.get("mean_ms", 0.0)),
                    "stdev_ms": float(timing.get("stdev_ms", 0.0)),
                    "measurement_scope": module.get("measurement_scope", ""),
                    "run_id": run.info.run_id,
                })
    return rows


def _render_modules(rows: list[dict[str, object]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6))
    if not rows:
        axis.text(0.5, 0.5, "No module measurements", ha="center", va="center")
        axis.set_axis_off()
    else:
        conditions = sorted({str(row["condition"]) for row in rows})
        components = sorted({str(row["component"]) for row in rows})
        bottoms = [0.0] * len(conditions)
        for index, component in enumerate(components):
            values = [
                next(
                    (
                        float(row["mean_ms"])
                        for row in rows
                        if row["condition"] == condition
                        and row["component"] == component
                    ),
                    0.0,
                )
                for condition in conditions
            ]
            axis.bar(
                range(len(conditions)), values, bottom=bottoms,
                label=component, color=ACCENT_COLORS[index % len(ACCENT_COLORS)],
            )
            bottoms = [left + right for left, right in zip(bottoms, values)]
        axis.set_xticks(
            range(len(conditions)),
            [condition.removeprefix("PF-W2V-") for condition in conditions],
            rotation=55,
            ha="right",
        )
        axis.set_ylabel("Component time (ms)")
        axis.legend(fontsize=8, ncol=2)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


__all__ = ["render"]
