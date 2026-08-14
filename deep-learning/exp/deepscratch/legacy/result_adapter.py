"""Historical MLflow projection; no canonical writer imports this module."""

from __future__ import annotations

from collections.abc import Iterable
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from mlflow.exceptions import MlflowException

from exp.framework.paths import WorkspacePaths
from exp.framework.results import MetricSeries, MlflowResultStore, NativeRunResult

from ..identity import Variant
from ..analysis.declarations import MetricDeclaration


def load_legacy_result(
    client,
    run_id: str,
    *,
    variant: Variant,
    declarations: Iterable[MetricDeclaration],
) -> NativeRunResult:
    """Project only retired namespace layouts into the native contract."""
    specs = []
    seen = set()
    for declaration in declarations:
        for metric_id in declaration.native_ids(variant):
            if metric_id not in seen:
                seen.add(metric_id)
                specs.append((metric_id, declaration.unit, declaration.split, declaration.axis))
    result = MlflowResultStore(client).load(
        run_id, metric_specs=specs, include_artifacts=False
    )
    missing = [item for item in declarations if not any(
        result.metric(metric_id) is not None for metric_id in item.native_ids(variant)
    )]
    aliases = _artifact_aliases(client, run_id)
    if not missing:
        return replace(result, artifact_aliases=aliases)
    rows = _raw_rows(client, run_id)
    projected = list(result.metrics)
    for declaration in missing:
        values = _project_rows(rows, declaration.metric_id)
        if not values and declaration.metric_id == "training_time_s":
            runtime = _runtime_projection(client, run_id)
            values = [] if runtime is None else [(0, runtime)]
        if values:
            native_id = declaration.native_ids(variant)[0]
            projected.append(MetricSeries(
                native_id,
                declaration.unit,
                declaration.split,
                declaration.axis,
                tuple(step for step, _ in values),
                tuple(value for _, value in values),
            ))
    return replace(
        result,
        metrics=tuple(projected),
        artifact_aliases=aliases,
    )


def _artifact_aliases(client, run_id: str) -> dict[str, str]:
    root_paths = {item.path for item in client.list_artifacts(run_id, "")}
    observation_paths = (
        {item.path for item in client.list_artifacts(run_id, "observations")}
        if "observations" in root_paths
        else set()
    )
    aliases = {}
    fallbacks = {
        "updates.csv": "raw/metrics.csv",
        "evaluations.csv": "raw/metrics.csv",
        "observations/source_curves.csv": "raw/metrics.csv",
        "observations/trajectory.csv": "raw/trajectory.csv",
        "observations/attention.csv": "raw/attention.csv",
        "observations/attention_render.json": "raw/attention_render.json",
    }
    for canonical, legacy in fallbacks.items():
        inventory = observation_paths if canonical.startswith("observations/") else root_paths
        if canonical not in inventory:
            aliases[canonical] = legacy
    if "observations/source_curves.csv" in aliases:
        source_curve = _source_curve_projection(client, run_id)
        if source_curve is not None:
            aliases["observations/source_curves.csv"] = str(source_curve)
    if "evaluations.csv" in aliases:
        evaluations = _evaluation_projection(client, run_id)
        if evaluations is not None:
            aliases["evaluations.csv"] = str(evaluations)
    run = client.get_run(run_id)
    study_id = (
        run.data.tags.get("experiment.id")
        or run.data.tags.get("experiment.ids", "").split(",")[0]
    )
    if (
        study_id == "e10"
        and "observations/activation_histogram.csv" not in observation_paths
    ):
        histogram = _activation_histogram_projection(client, run_id)
        if histogram is not None:
            aliases["observations/activation_histogram.csv"] = str(histogram)
    raw_paths = (
        {item.path for item in client.list_artifacts(run_id, "raw")}
        if "raw" in root_paths
        else set()
    )
    model_paths = (
        {item.path for item in client.list_artifacts(run_id, "model")}
        if "model" in root_paths
        else set()
    )
    if (
        "model/parameter_manifest.json" not in model_paths
        and "raw/parameter_manifest.json" in raw_paths
    ):
        parameter_manifest = _parameter_manifest_projection(client, run_id)
        if parameter_manifest is not None:
            aliases["model/parameter_manifest.json"] = str(parameter_manifest)
    if "checkpoints" in root_paths:
        aliases.update(_checkpoint_generation_aliases(client, run_id))
    return aliases


def _runtime_projection(client, run_id: str) -> float | None:
    try:
        path = Path(client.download_artifacts(run_id, "raw/timing.json"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = float(payload["training_wall_time_s"])
    except Exception:
        return None
    return value if np.isfinite(value) else None


def _parameter_manifest_projection(client, run_id: str) -> Path | None:
    target = (
        WorkspacePaths.from_environment(Path.cwd()).cache_root
        / "deepscratch"
        / "legacy-projections"
        / run_id
        / "parameter_manifest.json"
    )
    if target.is_file():
        return target.resolve()
    try:
        source = Path(
            client.download_artifacts(run_id, "raw/parameter_manifest.json")
        )
        payload = json.loads(source.read_text(encoding="utf-8"))
        count = int(payload["parameter_count"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps([{"name": "legacy_total", "numel": count}]) + "\n",
        encoding="utf-8",
    )
    return target.resolve()


def _evaluation_projection(client, run_id: str) -> Path | None:
    target = (
        WorkspacePaths.from_environment(Path.cwd()).cache_root
        / "deepscratch"
        / "legacy-projections"
        / run_id
        / "evaluations.csv"
    )
    if target.is_file():
        return target.resolve()
    rows = _raw_rows(client, run_id)
    if not rows or "perplexity" not in rows[0] or "epoch" not in rows[0]:
        return None
    projected = [
        {
            "axis_step": row["epoch"],
            "axis": "epoch",
            "split": row.get("split", "valid"),
            "metric": "perplexity",
            "value": row["perplexity"],
        }
        for row in rows
        if row.get("perplexity") not in {None, ""}
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("axis_step", "axis", "split", "metric", "value"),
        )
        writer.writeheader()
        writer.writerows(projected)
    return target.resolve()


def _source_curve_projection(client, run_id: str) -> Path | None:
    target = (
        WorkspacePaths.from_environment(Path.cwd()).cache_root
        / "deepscratch"
        / "legacy-projections"
        / run_id
        / "source_curves.csv"
    )
    if target.is_file():
        return target.resolve()
    rows = _raw_rows(client, run_id)
    if not rows:
        return None
    columns = (
        ("book_loss", "loss"),
        ("perplexity", "perplexity"),
        ("exact_match_accuracy", "exact_match"),
        ("exact_match_accuracy", "accuracy"),
    )
    metric, value_column = next(
        ((metric, column) for metric, column in columns if column in rows[0]),
        (None, None),
    )
    if metric is None or value_column is None:
        return None
    projected = []
    for index, row in enumerate(rows):
        value = row.get(value_column)
        if value in {None, ""}:
            continue
        projected.append({
            "plot_index": row.get("plot_index") or row.get("epoch") or index,
            "metric": metric,
            "value": value,
        })
    if not projected:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("plot_index", "metric", "value"))
        writer.writeheader()
        writer.writerows(projected)
    return target.resolve()


def _activation_histogram_projection(client, run_id: str) -> Path | None:
    target = (
        WorkspacePaths.from_environment(Path.cwd()).cache_root
        / "deepscratch"
        / "legacy-projections"
        / run_id
        / "activation_histogram.csv"
    )
    if target.is_file():
        return target.resolve()
    try:
        source = Path(client.download_artifacts(run_id, "raw/activations.npz"))
        with np.load(source, allow_pickle=False) as arrays:
            rows = []
            for layer in range(1, 6):
                counts, edges = np.histogram(
                    np.asarray(arrays[f"layer_{layer}"]).ravel(),
                    bins=30,
                    range=(0.0, 1.0),
                )
                rows.extend(
                    {
                        "layer": layer,
                        "bin_index": index,
                        "bin_left": float(edges[index]),
                        "bin_right": float(edges[index + 1]),
                        "count": int(count),
                    }
                    for index, count in enumerate(counts)
                )
    except (KeyError, OSError, ValueError, MlflowException):
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("layer", "bin_index", "bin_left", "bin_right", "count"),
        )
        writer.writeheader()
        writer.writerows(rows)
    return target.resolve()


def _checkpoint_generation_aliases(client, run_id: str) -> dict[str, str]:
    try:
        manifest_path = Path(
            client.download_artifacts(
                run_id,
                "checkpoints/checkpoint_manifest.json",
            )
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        final = manifest.get("final")
        if not isinstance(final, dict) or not final.get("path"):
            return {}
        generation = Path(str(final["path"])).name
    except Exception:
        return {}
    return {
        f"checkpoints/{generation}/model_parameters.npz": (
            f"checkpoints/generations/{generation}/model_parameters.npz"
        )
    }


def _raw_rows(client, run_id: str) -> list[dict[str, str]]:
    for artifact in ("raw/metrics.csv", "metrics.csv", "metrics/metrics.csv"):
        try:
            path = Path(client.download_artifacts(run_id, artifact))
        except Exception:
            continue
        if path.is_file():
            with path.open(encoding="utf-8", newline="") as stream:
                return list(csv.DictReader(stream))
    return []


def _project_rows(
    rows: list[dict[str, str]],
    metric_id: str,
) -> list[tuple[int, float]]:
    output = []
    for index, row in enumerate(rows):
        metric = str(row.get("metric", "")).lower()
        split = str(row.get("split", "")).lower()
        candidates: tuple[str, ...]
        if metric_id == "test_accuracy":
            if split not in {"test", "test-full"}:
                continue
            candidates = ("accuracy", "value")
        elif metric_id == "train_loss":
            if split and split != "train":
                continue
            if metric and not any(word in metric for word in ("loss", "objective")):
                continue
            candidates = ("loss", "objective", "value")
        elif metric_id == "train_perplexity":
            if split and split != "train":
                continue
            candidates = ("perplexity", "value")
        elif metric_id == "test_perplexity":
            if split not in {"", "test"}:
                continue
            candidates = ("perplexity", "value")
        elif metric_id == "test_exact_match":
            if split not in {"", "test", "test-full"}:
                continue
            candidates = ("exact_match", "accuracy", "value")
        elif metric_id == "train_accuracy_curve":
            if split not in {"", "train"}:
                continue
            candidates = ("accuracy", "value")
        elif metric_id == "test_accuracy_curve":
            if split not in {"test", "test-full"}:
                continue
            candidates = ("accuracy", "value")
        else:
            continue
        value = next((row.get(key) for key in candidates if row.get(key) not in {None, ""}), None)
        try:
            number = float(value) if value is not None else None
        except ValueError:
            continue
        if number is None:
            continue
        step_value = (
            row.get("epoch")
            if metric_id in {"train_accuracy_curve", "test_accuracy_curve"}
            else None
        ) or row.get("update") or row.get("epoch") or row.get("plot_index")
        try:
            step = int(float(step_value)) if step_value not in {None, ""} else index
        except ValueError:
            step = index
        output.append((step, number))
    return output
