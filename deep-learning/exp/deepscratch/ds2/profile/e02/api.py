"""DS2 e02 Word2Vec profile orchestration."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from exp.deepscratch.ds2.profile.paths import profile_cache

from .modules import COMPONENTS, profile_modules
from .update import (
    CONDITIONS,
    _load_data,
    _metadata,
    profile_condition,
)


DEFAULT_RESULTS = profile_cache("e02")


def run(
    *,
    devices: tuple[str, ...] = ("cpu", "cuda:0"),
    conditions: tuple[str, ...] | None = None,
    mode: str = "all",
    components: tuple[str, ...] | None = None,
    batch_size: int = 100,
    epochs: int = 10,
    update_warmup: int = 20,
    update_repetitions: int = 5,
    measured_updates: int = 50,
    module_warmup: int = 5,
    module_iterations: int = 20,
    output_dir: Path = DEFAULT_RESULTS,
) -> None:
    """Run the selected e02 profiles, or the complete CPU/GPU matrix."""
    if mode not in {"all", "update", "modules"}:
        raise ValueError(f"unknown profile mode: {mode}")
    selected_conditions = CONDITIONS if conditions is None else conditions
    unknown_conditions = set(selected_conditions) - set(CONDITIONS)
    if unknown_conditions:
        raise ValueError(
            f"unknown e02 profile conditions: {sorted(unknown_conditions)}"
        )
    unknown_components = set(components or ()) - set(COMPONENTS)
    if unknown_components:
        raise ValueError(
            f"unknown e02 profile components: {sorted(unknown_components)}"
        )

    collected: dict[
        str,
        tuple[list[dict[str, object]], list[dict[str, object]]],
    ] = {}
    for device in devices:
        backend, corpus, contexts, targets = _load_data(device)
        device_dir = output_dir / device.replace(":", "")
        device_dir.mkdir(parents=True, exist_ok=True)
        update_rows: list[dict[str, object]] = []
        module_rows: list[dict[str, object]] = []

        if mode in {"all", "update"}:
            for condition in selected_conditions:
                backend.seed(1)
                np.random.seed(1)
                result = profile_condition(
                    condition,
                    corpus=corpus,
                    contexts=contexts,
                    targets=targets,
                    backend=backend,
                    batch_size=batch_size,
                    epochs=epochs,
                    warmup_updates=update_warmup,
                    measured_updates=measured_updates,
                    phase_updates=0,
                    repetitions=update_repetitions,
                )
                update_rows.append(asdict(result))
                print(
                    f"[{device}] {condition} update: "
                    f"cold {result.cold_ms_per_update:.3f} ms; "
                    f"steady {result.mean_ms_per_update:.3f} ± "
                    f"{result.stdev_ms_per_update:.3f} ms; "
                    f"{result.estimated_seconds_per_epoch:.1f} ± "
                    f"{result.estimated_repeat_stdev_seconds_per_epoch:.1f} "
                    f"s/epoch; {result.estimated_seconds_total:.1f} ± "
                    f"{result.estimated_repeat_stdev_seconds_total:.1f} s total",
                    flush=True,
                )
            _write_payload(
                device_dir / "update.json",
                backend=backend,
                stage="update",
                results=update_rows,
                schema_version=6,
            )

        if mode in {"all", "modules"}:
            for condition in selected_conditions:
                backend.seed(1)
                np.random.seed(1)
                rows = profile_modules(
                    condition,
                    corpus=corpus,
                    contexts=contexts,
                    targets=targets,
                    backend=backend,
                    batch_size=batch_size,
                    components=components,
                    warmup_iterations=module_warmup,
                    measured_iterations=module_iterations,
                )
                module_rows.extend(rows)
                for row in rows:
                    timing = row["timing"]
                    assert isinstance(timing, dict)
                    print(
                        f"[{device}] {condition} {row['component']}: "
                        f"{float(timing['mean_ms']):.3f} ± "
                        f"{float(timing['stdev_ms']):.3f} ms",
                        flush=True,
                    )
            _write_payload(
                device_dir / "modules.json",
                backend=backend,
                stage="modules",
                results=module_rows,
                schema_version=1,
            )
        collected[device] = (update_rows, module_rows)

    print("\n# DS2 e02 profiling summary", flush=True)
    selected_models = tuple(
        model
        for model in ("CBOW", "SkipGram")
        if any(f"-{model.lower()}-" in condition for condition in selected_conditions)
    )
    for device, (update_rows, module_rows) in collected.items():
        for model in selected_models:
            print(
                "\n"
                + render_summary_table(
                    device=device,
                    model=model,
                    update_rows=update_rows,
                    module_rows=module_rows,
                ),
                flush=True,
            )


def _write_payload(
    path: Path,
    *,
    backend,
    stage: str,
    results: list[dict[str, object]],
    schema_version: int,
) -> None:
    payload = {
        "schema_version": schema_version,
        "metadata": _metadata(backend, stage=stage),
        "results": results,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved: {path}", flush=True)


SUMMARY_COLUMNS = (
    ("Original One-hot FS", "original", "onehot-fs"),
    ("Original Emb. FS", "original", "fs"),
    ("Original NS", "original", "ns"),
    ("Implemented One-hot FS", "implemented", "onehot-fs"),
    ("Implemented Emb. FS", "implemented", "fs"),
    ("Implemented NS", "implemented", "ns"),
    ("Implemented Fused NS", "implemented", "fused-ns"),
)
SUMMARY_ROWS = (
    ("Cold update", "cold_update", "ms"),
    ("Update", "update", "ms"),
    ("Steady event p50", "steady_event_p50", "ms"),
    ("Steady event p95", "steady_event_p95", "ms"),
    ("Epoch", "epoch", "s"),
    ("Total", "total", "s"),
    ("Batch adapter", "batch_adapter", "ms"),
    ("Objective prepare", "objective_prepare", "ms"),
    ("Model forward", "model_forward", "ms"),
    ("Objective forward", "objective_forward", "ms"),
    ("Objective backward", "objective_backward", "ms"),
    ("Model backward", "model_backward", "ms"),
    ("Fused forward + loss", "fused_forward_loss", "ms"),
    ("Fused backward", "fused_backward", "ms"),
    (
        "Deduplicate shared parameters",
        "deduplicate_shared_parameters",
        "ms",
    ),
    ("Optimizer", "optimizer", "ms"),
)
BOLD_FASTEST_ROWS = {
    "update",
    "epoch",
    "total",
    "model_forward",
    "model_backward",
    "fused_forward_loss",
    "fused_backward",
    "optimizer",
}


def render_summary_table(
    *,
    device: str,
    model: str,
    update_rows: list[dict[str, object]],
    module_rows: list[dict[str, object]],
) -> str:
    """Render one model/device comparison table after profiling completes."""
    model_token = model.lower()
    update_by_condition = {str(row["condition"]): row for row in update_rows}
    modules_by_key = {
        (str(row["condition"]), str(row["component"])): row for row in module_rows
    }
    rendered_rows: list[list[str]] = []
    for label, metric, unit in SUMMARY_ROWS:
        values = [
            _summary_value(
                condition=_condition_name(
                    model_token,
                    implementation,
                    objective,
                ),
                metric=metric,
                update_by_condition=update_by_condition,
                modules_by_key=modules_by_key,
            )
            for _column, implementation, objective in SUMMARY_COLUMNS
        ]
        fastest = (
            min(value[0] for value in values if value is not None)
            if metric in BOLD_FASTEST_ROWS and any(values)
            else None
        )
        cells = [
            _format_summary_value(
                value,
                unit=unit,
                bold=(
                    fastest is not None and value is not None and value[0] == fastest
                ),
            )
            for value in values
        ]
        rendered_rows.append([f"**{label}**" if metric == "update" else label, *cells])

    headers = ["Metric", *(column[0] for column in SUMMARY_COLUMNS)]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rendered_rows))
        for index in range(len(headers))
    ]
    lines = [
        f"## {device} · {model}",
        "",
        _markdown_row(headers, widths, right_aligned=False),
        "| "
        + " | ".join(
            "-" * widths[0] if index == 0 else "-" * max(3, widths[index] - 1) + ":"
            for index in range(len(widths))
        )
        + " |",
    ]
    lines.extend(
        _markdown_row(row, widths, right_aligned=True) for row in rendered_rows
    )
    lines.extend(
        [
            "",
            "¹ Original forward/backward include the objective because the "
            "book model does not expose separate loss boundaries.",
            "All ± values are standard deviations across repeated throughput "
            "windows; epoch and total values use the same linear extrapolation.",
        ]
    )
    return "\n".join(lines)


def _condition_name(
    model: str,
    implementation: str,
    objective: str,
) -> str:
    return f"{implementation}-{model}-{objective}"


def _summary_value(
    *,
    condition: str,
    metric: str,
    update_by_condition: dict[str, dict[str, object]],
    modules_by_key: dict[tuple[str, str], dict[str, object]],
) -> tuple[float, float | None, bool] | None:
    if metric in {
        "cold_update",
        "update",
        "steady_event_p50",
        "steady_event_p95",
        "epoch",
        "total",
    }:
        row = update_by_condition.get(condition)
        if row is None:
            return None
        fields = {
            "cold_update": ("cold_ms_per_update", None),
            "update": ("mean_ms_per_update", "stdev_ms_per_update"),
            "steady_event_p50": ("steady_event_p50_ms_per_update", None),
            "steady_event_p95": ("steady_event_p95_ms_per_update", None),
            "epoch": (
                "estimated_seconds_per_epoch",
                "estimated_repeat_stdev_seconds_per_epoch",
            ),
            "total": (
                "estimated_seconds_total",
                "estimated_repeat_stdev_seconds_total",
            ),
        }
        mean_field, stdev_field = fields[metric]
        if mean_field not in row:
            return None
        spread_value = (
            None
            if stdev_field is None or row.get(stdev_field) is None
            else float(row[stdev_field])
        )
        return float(row[mean_field]), spread_value, False

    component = metric
    combined_original = False
    if condition.startswith("original-"):
        component = {
            "model_forward": "forward",
            "model_backward": "backward",
        }.get(metric, metric)
        combined_original = metric in {"model_forward", "model_backward"}
    row = modules_by_key.get((condition, component))
    if row is None:
        return None
    timing = row["timing"]
    assert isinstance(timing, dict)
    return (
        float(timing["mean_ms"]),
        float(timing["stdev_ms"]),
        combined_original,
    )


def _format_summary_value(
    value: tuple[float, float | None, bool] | None,
    *,
    unit: str,
    bold: bool,
) -> str:
    if value is None:
        return "—"
    mean_value, repeat_stdev, footnote = value
    precision = 3 if unit == "ms" else 1
    rendered = f"{mean_value:.{precision}f} {unit}"
    if repeat_stdev is not None:
        rendered += f" ± {repeat_stdev:.{precision}f}"
    if bold:
        rendered = f"**{rendered}**"
    return rendered + ("¹" if footnote else "")


def _markdown_row(
    cells: list[str],
    widths: list[int],
    *,
    right_aligned: bool,
) -> str:
    formatted = []
    for index, (cell, width) in enumerate(zip(cells, widths, strict=True)):
        if right_aligned and index > 0:
            formatted.append(cell.rjust(width))
        else:
            formatted.append(cell.ljust(width))
    return "| " + " | ".join(formatted) + " |"
