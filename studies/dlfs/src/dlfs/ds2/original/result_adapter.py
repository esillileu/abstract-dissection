"""Adapter for promoted-original DS2 SchemaV1 runs."""

import csv
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from mlflow.exceptions import MlflowException

from dlfs.identity import Variant
from repro_core.context.paths import WorkspacePaths
from repro_core.results import MlflowResultStore
from repro_mlflow.artifact_cache import MlflowArtifactCache


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
    if (
        study_id in {"e02", "e12"}
        and "checkpoints/checkpoint_manifest.json" not in aliases
    ):
        manifest = _word2vec_checkpoint_projection(
            client, run_id, artifact_cache=artifact_cache
        )
        if manifest is not None:
            aliases["checkpoints/checkpoint_manifest.json"] = str(manifest)
    if study_id == "e08":
        attention = _attention_projection(client, run_id, artifact_cache=artifact_cache)
        if attention is not None:
            attention_csv, attention_render = attention
            aliases["observations/attention.csv"] = str(attention_csv)
            aliases["observations/attention_render.json"] = str(attention_render)
    return replace(result, artifact_aliases=aliases)


def _attention_projection(
    client, run_id: str, *, artifact_cache: MlflowArtifactCache | None = None
) -> tuple[Path, Path] | None:
    root = (
        WorkspacePaths.from_environment(Path.cwd()).cache_root
        / "deepscratch"
        / "legacy-projections"
        / run_id
        / "attention"
    )
    csv_path = root / "attention.csv"
    render_path = root / "attention_render.json"
    if csv_path.is_file() and render_path.is_file():
        return csv_path.resolve(), render_path.resolve()
    try:
        tracking_uri = (
            getattr(client, "tracking_uri", None)
            or os.getenv("REPRO_TRACKING_URI")
            or os.getenv("MLFLOW_TRACKING_URI")
            or os.getenv("MLFLOW_F1_URL")
            or "http://127.0.0.1:5000"
        )
        cache = artifact_cache or MlflowArtifactCache(client, str(tracking_uri))
        arrays_path = cache.get(run_id, "raw/attention.npz")
        labels_path = cache.get(run_id, "raw/labels.csv")
        with labels_path.open(newline="", encoding="utf-8") as stream:
            labels = list(csv.DictReader(stream))
        with np.load(arrays_path, allow_pickle=False) as archive:
            projected = []
            render = {}
            for index, label in enumerate(labels):
                values = np.asarray(archive[f"attention_{index}"])
                example_id = str(label.get("dataset_index") or label["example"])
                render[example_id] = {
                    "source_labels": list(label["row_labels"]),
                    "target_labels": list(label["column_labels"]),
                }
                projected.extend(
                    {
                        "example_id": example_id,
                        "decode_step": decode_step,
                        "encoder_position": encoder_position,
                        "weight": float(values[decode_step, encoder_position]),
                    }
                    for decode_step in range(values.shape[0])
                    for encoder_position in range(values.shape[1])
                )
    except (KeyError, OSError, ValueError, MlflowException):
        return None
    root.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("example_id", "decode_step", "encoder_position", "weight"),
        )
        writer.writeheader()
        writer.writerows(projected)
    render_path.write_text(
        json.dumps({"examples": render}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path.resolve(), render_path.resolve()


def _word2vec_checkpoint_projection(
    client, run_id: str, *, artifact_cache: MlflowArtifactCache | None = None
) -> Path | None:
    """Expose promoted-original e02 ``word_vectors`` as canonical ``W_in``."""
    root = (
        WorkspacePaths.from_environment(Path.cwd()).cache_root
        / "deepscratch"
        / "legacy-projections"
        / run_id
        / "checkpoint"
    )
    manifest_path = root / "checkpoint_manifest.json"
    weights_path = root / "model_parameters.npz"
    if manifest_path.is_file() and weights_path.is_file():
        return manifest_path.resolve()
    try:
        tracking_uri = (
            getattr(client, "tracking_uri", None)
            or os.getenv("REPRO_TRACKING_URI")
            or os.getenv("MLFLOW_TRACKING_URI")
            or os.getenv("MLFLOW_F1_URL")
            or "http://127.0.0.1:5000"
        )
        cache = artifact_cache or MlflowArtifactCache(client, str(tracking_uri))
        source = cache.get(run_id, "raw/checkpoint.npz")
        with np.load(source, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        vectors = arrays.get("W_in", arrays.get("word_vectors"))
        if vectors is None or vectors.ndim != 2:
            return None
    except (OSError, ValueError, MlflowException):
        return None
    root.mkdir(parents=True, exist_ok=True)
    arrays.setdefault("W_in", vectors)
    np.savez_compressed(weights_path, **arrays)
    manifest_path.write_text(
        json.dumps(
            {
                "format": "promoted-original-word2vec-npz",
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
    return list(
        dict.fromkeys(
            (metric_id, declaration.unit, declaration.split, declaration.axis)
            for declaration in declarations
            for metric_id in declaration.native_ids(Variant.ORIGINAL)
        )
    )
