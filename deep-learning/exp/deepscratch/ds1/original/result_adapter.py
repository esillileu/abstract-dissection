"""Adapter for promoted-original DS1 SchemaV1 runs."""

import csv
from dataclasses import replace
import json
import os
from pathlib import Path

from mlflow.exceptions import MlflowException
import numpy as np

from exp.deepscratch.identity import Variant
from exp.framework.paths import WorkspacePaths
from exp.framework.results import MlflowResultStore
from mlprosection_mlflow.artifact_cache import MlflowArtifactCache


def load_native_result(client, run_id, declarations, *, artifact_cache=None):
    specs = _metric_specs(declarations)
    result = MlflowResultStore(client).load(
        run_id, metric_specs=specs, include_artifacts=False
    )
    if result.schema_version != 1:
        raise ValueError(
            f"canonical run {run_id} has unsupported schema version "
            f"{result.schema_version}"
        )
    aliases = dict(result.artifact_aliases)
    study_id = result.provenance.get("experiment_id")
    cache = artifact_cache or MlflowArtifactCache(
        client,
        str(
            getattr(client, "tracking_uri", None)
            or os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        ),
    )
    if study_id == "e09":
        trajectory = _cached_artifact(cache, run_id, "raw/trajectory.csv")
        if trajectory is not None:
            aliases["observations/trajectory.csv"] = str(trajectory)
    elif study_id == "e10":
        activation = _activation_projection(cache, run_id)
        if activation is not None:
            aliases["observations/activation_histogram.csv"] = str(activation)
    elif study_id == "e06":
        checkpoint = _checkpoint_projection(cache, run_id)
        if checkpoint is not None:
            aliases["checkpoints/checkpoint_manifest.json"] = str(checkpoint)
    return replace(result, artifact_aliases=aliases)


def _cached_artifact(cache, run_id: str, path: str) -> Path | None:
    try:
        return cache.get(run_id, path)
    except (OSError, MlflowException):
        return None


def _activation_projection(cache, run_id: str) -> Path | None:
    source = _cached_artifact(cache, run_id, "raw/activations.npz")
    if source is None:
        return None
    root = (
        WorkspacePaths.from_environment(Path.cwd()).cache_root
        / "deepscratch"
        / "legacy-projections"
        / run_id
        / "activation"
    )
    output = root / "activation_histogram.csv"
    if output.is_file():
        return output.resolve()
    try:
        with np.load(source, allow_pickle=False) as archive:
            rows = []
            for layer in range(1, 6):
                values = np.asarray(archive[f"layer_{layer}"]).ravel()
                counts, edges = np.histogram(values, bins=30, range=(0.0, 1.0))
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
    except (KeyError, OSError, ValueError):
        return None
    root.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("layer", "bin_index", "bin_left", "bin_right", "count"),
        )
        writer.writeheader()
        writer.writerows(rows)
    return output.resolve()


def _checkpoint_projection(cache, run_id: str) -> Path | None:
    source = _cached_artifact(cache, run_id, "raw/checkpoint.npz")
    if source is None:
        return None
    root = (
        WorkspacePaths.from_environment(Path.cwd()).cache_root
        / "deepscratch"
        / "legacy-projections"
        / run_id
        / "checkpoint"
    )
    weights_path = root / "model_parameters.npz"
    manifest_path = root / "checkpoint_manifest.json"
    if weights_path.is_file() and manifest_path.is_file():
        return manifest_path.resolve()
    try:
        with np.load(source, allow_pickle=False) as archive:
            arrays = {
                (
                    f"layer.{name.removeprefix('param__')[1:]}.W"
                    if name.removeprefix("param__").startswith("W")
                    else f"layer.{name.removeprefix('param__')[1:]}.b"
                ): np.asarray(archive[name])
                for name in archive.files
                if name.startswith("param__")
            }
    except (OSError, ValueError):
        return None
    if not arrays:
        return None
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(weights_path, **arrays)
    manifest_path.write_text(
        json.dumps(
            {
                "format": "promoted-original-ds1-npz",
                "final": {
                    "path": str(weights_path.resolve()),
                    "epoch": "",
                    "update": "",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path.resolve()


def _metric_specs(declarations):
    return list(dict.fromkeys(
        (metric_id, declaration.unit, declaration.split, declaration.axis)
        for declaration in declarations
        for metric_id in declaration.native_ids(Variant.ORIGINAL)
    ))
