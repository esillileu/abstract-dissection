"""Plot the measured GPU module-time breakdown for DS2 e02 Word2Vec."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exp.framework.plotting.theme import ACCENT_COLORS, INK, MUTED, apply_plot_theme
from exp.deepscratch.ds2.profile.paths import profile_artifacts, profile_cache

CONDITIONS = (
    "implemented-cbow-fs",
    "implemented-cbow-ns",
    "implemented-cbow-fused-ns",
    "implemented-skipgram-fs",
    "implemented-skipgram-ns",
    "implemented-skipgram-fused-ns",
)
DEFAULT_RESULTS_DIR = profile_cache("e02")
DEFAULT_OUTPUT_DIR = profile_artifacts("e02")
OUTPUT_STEM = "gpu_module_time_stacked"
CSV_FIELDS = (
    "model",
    "method",
    "prepare_ms",
    "forward_loss_ms",
    "backward_ms",
    "optimizer_ms",
    "other_ms",
    "update_mean_ms",
    "update_std_ms",
)

_REGULAR_COMPONENTS = {
    "prepare_ms": ("batch_adapter", "objective_prepare"),
    "forward_loss_ms": ("model_forward", "objective_forward"),
    "backward_ms": ("objective_backward", "model_backward"),
    "optimizer_ms": ("optimizer",),
}
_FUSED_COMPONENTS = {
    "prepare_ms": ("batch_adapter", "objective_prepare"),
    "forward_loss_ms": ("fused_forward_loss",),
    "backward_ms": ("fused_backward",),
    "optimizer_ms": ("optimizer",),
}
_OBJECTIVE_METHODS = {
    "FullSoftmax": "Full Softmax",
    "NegativeSampling": "Negative Sampling",
    "FusedNegativeSampling": "Fused Negative Sampling",
}
_PLOT_METHODS = {
    "Full Softmax": "Full Softmax",
    "Negative Sampling": "Negative Sampling",
    "Fused Negative Sampling": "Fused NS",
}
_METADATA_KEYS = (
    "backend",
    "cuda_runtime_version",
    "cupy_version",
    "device",
    "device_name",
    "method",
    "numpy_version",
)


@dataclass(frozen=True)
class ModuleBreakdown:
    model: str
    method: str
    prepare_ms: float
    forward_loss_ms: float
    backward_ms: float
    optimizer_ms: float
    other_ms: float
    update_mean_ms: float
    update_std_ms: float

    @property
    def stacked_total_ms(self) -> float:
        return (
            self.prepare_ms
            + self.forward_loss_ms
            + self.backward_ms
            + self.optimizer_ms
            + self.other_ms
        )


def load_payload(path: Path) -> dict[str, object]:
    """Load one profiler JSON payload without changing or remeasuring it."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"profiling input does not exist: {path}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"invalid profiling payload in {path}: missing results list")
    return payload


def _validate_metadata(
    update_payload: dict[str, object],
    modules_payload: dict[str, object],
) -> None:
    update_metadata = update_payload.get("metadata")
    module_metadata = modules_payload.get("metadata")
    if not isinstance(update_metadata, dict) or not isinstance(module_metadata, dict):
        return
    mismatches = [
        f"{key}: update={update_metadata[key]!r}, modules={module_metadata[key]!r}"
        for key in _METADATA_KEYS
        if key in update_metadata
        and key in module_metadata
        and update_metadata[key] != module_metadata[key]
    ]
    if mismatches:
        raise ValueError(
            "update.json and modules.json profiling metadata differ: "
            + "; ".join(mismatches)
        )


def _required_float(row: dict[str, object], field: str, condition: str) -> float:
    if field not in row:
        raise ValueError(f"{condition}: missing required field '{field}'")
    try:
        value = float(row[field])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{condition}: field '{field}' is not numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"{condition}: field '{field}' must be finite")
    return value


def _module_mean(
    modules_by_key: dict[tuple[str, str], dict[str, object]],
    condition: str,
    component: str,
) -> float:
    row = modules_by_key.get((condition, component))
    if row is None:
        raise ValueError(
            f"{condition}: missing required module component '{component}'"
        )
    timing = row.get("timing")
    if not isinstance(timing, dict):
        raise ValueError(f"{condition}.{component}: missing required field 'timing'")
    return _required_float(timing, "mean_ms", f"{condition}.{component}")


def aggregate_module_breakdowns(
    update_payload: dict[str, object],
    modules_payload: dict[str, object],
) -> list[ModuleBreakdown]:
    """Aggregate profiler means into mutually exclusive ms/update sections."""
    _validate_metadata(update_payload, modules_payload)
    update_results = update_payload.get("results")
    module_results = modules_payload.get("results")
    if not isinstance(update_results, list) or not isinstance(module_results, list):
        raise ValueError("profiling payloads must contain results lists")

    updates = {
        str(row["condition"]): row
        for row in update_results
        if isinstance(row, dict) and "condition" in row
    }
    modules_by_key = {
        (str(row["condition"]), str(row["component"])): row
        for row in module_results
        if isinstance(row, dict) and "condition" in row and "component" in row
    }
    breakdowns = []
    for condition in CONDITIONS:
        update_row = updates.get(condition)
        if update_row is None:
            raise ValueError(f"{condition}: missing required update result")
        objective = update_row.get("objective")
        if objective not in _OBJECTIVE_METHODS:
            raise ValueError(
                f"{condition}: missing or unsupported required field 'objective'"
            )
        model = update_row.get("model")
        if model not in {"CBOW", "SkipGram"}:
            raise ValueError(f"{condition}: missing or unsupported required field 'model'")
        if update_row.get("implementation") != "implemented":
            raise ValueError(f"{condition}: required field 'implementation' is not implemented")

        component_map = (
            _FUSED_COMPONENTS
            if objective == "FusedNegativeSampling"
            else _REGULAR_COMPONENTS
        )
        expected_scope = (
            "fused_negative_sampling"
            if objective == "FusedNegativeSampling"
            else "separate_model_objective"
        )
        update_batch_size = update_row.get("batch_size")
        values: dict[str, float] = {}
        for aggregate_field, components in component_map.items():
            for component in components:
                module_row = modules_by_key.get((condition, component))
                if module_row is not None:
                    if module_row.get("batch_size") != update_batch_size:
                        raise ValueError(
                            f"{condition}.{component}: batch_size differs between "
                            "update.json and modules.json"
                        )
                    if module_row.get("model") != model or module_row.get(
                        "objective"
                    ) != objective:
                        raise ValueError(
                            f"{condition}.{component}: model/objective metadata "
                            "differs between update.json and modules.json"
                        )
                    if module_row.get("measurement_scope") != expected_scope:
                        raise ValueError(
                            f"{condition}.{component}: unexpected measurement_scope "
                            f"{module_row.get('measurement_scope')!r}"
                        )
            values[aggregate_field] = sum(
                _module_mean(modules_by_key, condition, component)
                for component in components
            )

        update_mean = _required_float(
            update_row, "mean_ms_per_update", condition
        )
        update_std = _required_float(
            update_row, "stdev_ms_per_update", condition
        )
        measured_sum = sum(values.values())
        other = update_mean - measured_sum
        if other < -0.05 or other < -0.05 * update_mean:
            raise ValueError(
                f"{condition}: Other is implausibly negative ({other:.6f} ms/update; "
                f"update mean {update_mean:.6f} ms/update). Check input pairing, "
                "units, and duplicate component aggregation."
            )
        if other < 0.0:
            if not math.isclose(other, 0.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    f"{condition}: Other is negative ({other:.9f} ms/update), "
                    "which exceeds floating-point tolerance"
                )
            other = 0.0

        breakdown = ModuleBreakdown(
            model=str(model),
            method=_OBJECTIVE_METHODS[str(objective)],
            other_ms=other,
            update_mean_ms=update_mean,
            update_std_ms=update_std,
            **values,
        )
        if not math.isclose(
            breakdown.stacked_total_ms,
            update_mean,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"{condition}: stacked total {breakdown.stacked_total_ms:.9f} "
                f"does not match update mean {update_mean:.9f} ms/update"
            )
        breakdowns.append(breakdown)
    return breakdowns


def write_breakdowns_csv(rows: list[ModuleBreakdown], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return path


def render_module_breakdowns(rows: list[ModuleBreakdown]):
    """Render CBOW and Skip-gram stacked means with update-level SD bars."""
    apply_plot_theme()
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=True)
    sections = (
        ("Prepare", "prepare_ms", ACCENT_COLORS[0]),
        ("Forward + loss", "forward_loss_ms", ACCENT_COLORS[1]),
        ("Backward", "backward_ms", ACCENT_COLORS[2]),
        ("Optimizer", "optimizer_ms", ACCENT_COLORS[3]),
        ("Other", "other_ms", ACCENT_COLORS[4]),
    )
    method_order = tuple(_PLOT_METHODS)
    ymax = max(row.update_mean_ms + row.update_std_ms for row in rows)
    label_offset = max(0.025 * ymax, 0.04)

    for axis, model in zip(axes, ("CBOW", "SkipGram"), strict=True):
        model_rows = {row.method: row for row in rows if row.model == model}
        ordered = [model_rows[method] for method in method_order]
        x = np.arange(len(ordered))
        bottoms = np.zeros(len(ordered))
        for label, field, color in sections:
            values = np.asarray([getattr(row, field) for row in ordered])
            axis.bar(
                x,
                values,
                bottom=bottoms,
                width=0.68,
                label=label,
                color=color,
                edgecolor=INK,
                linewidth=0.35,
            )
            bottoms += values
        means = np.asarray([row.update_mean_ms for row in ordered])
        standard_deviations = np.asarray([row.update_std_ms for row in ordered])
        axis.errorbar(
            x,
            means,
            yerr=standard_deviations,
            fmt="none",
            ecolor=INK,
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            zorder=5,
        )
        for index, row in enumerate(ordered):
            axis.text(
                index,
                row.update_mean_ms + row.update_std_ms + label_offset,
                f"{row.update_mean_ms:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color=INK,
            )
        axis.set_title("Skip-gram" if model == "SkipGram" else model, fontsize=13)
        axis.set_xticks(x, [_PLOT_METHODS[row.method] for row in ordered])
        axis.tick_params(axis="x", labelrotation=0, labelsize=9)
        axis.grid(axis="y", alpha=0.25, color=MUTED)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Update time (ms)", fontsize=11)
    axes[0].set_ylim(0.0, ymax + 3.2 * label_offset)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=True,
        bbox_to_anchor=(0.5, 1.0),
        fontsize=9,
    )
    figure.subplots_adjust(top=0.84, bottom=0.14, left=0.08, right=0.98, wspace=0.12)
    return figure


def generate(
    *,
    device: str = "cuda0",
    results_dir: Path = DEFAULT_RESULTS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[list[ModuleBreakdown], tuple[Path, Path, Path]]:
    """Read existing profile JSON and write CSV, 300 dpi PNG, and SVG."""
    device_dir = results_dir / device.replace(":", "")
    update_payload = load_payload(device_dir / "update.json")
    modules_payload = load_payload(device_dir / "modules.json")
    rows = aggregate_module_breakdowns(update_payload, modules_payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_breakdowns_csv(rows, output_dir / f"{OUTPUT_STEM}.csv")
    figure = render_module_breakdowns(rows)
    png_path = output_dir / f"{OUTPUT_STEM}.png"
    svg_path = output_dir / f"{OUTPUT_STEM}.svg"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    return rows, (png_path, svg_path, csv_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda0")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    rows, paths = generate(
        device=args.device,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )
    for row in rows:
        print(
            f"{row.model} {row.method}: {row.update_mean_ms:.3f} ± "
            f"{row.update_std_ms:.3f} ms/update"
        )
    for path in paths:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
