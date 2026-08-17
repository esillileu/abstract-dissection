"""Adapter for promoted-original DS2 SchemaV1 runs."""

import json
from dataclasses import replace
import os
from pathlib import Path

import numpy as np
from mlflow.exceptions import MlflowException

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
    if "checkpoints/checkpoint_manifest.json" not in aliases:
        manifest = _word2vec_checkpoint_projection(
            client, run_id, artifact_cache=artifact_cache
        )
        if manifest is not None:
            aliases["checkpoints/checkpoint_manifest.json"] = str(manifest)
    return replace(result, artifact_aliases=aliases)


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
        tracking_uri = getattr(client, "tracking_uri", None) or os.getenv(
            "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"
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
    return list(dict.fromkeys(
        (metric_id, declaration.unit, declaration.split, declaration.axis)
        for declaration in declarations
        for metric_id in declaration.native_ids(Variant.ORIGINAL)
    ))
