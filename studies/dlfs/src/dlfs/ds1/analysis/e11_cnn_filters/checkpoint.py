"""Checkpoint weight extraction and path resolution for CNN filters."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from dlfs.analysis.input import artifact_file


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)
    )


def _conv_weights(checkpoint: Path) -> list[tuple[str, np.ndarray]]:
    """Return convolution kernels in model order from a v1 or v2 checkpoint."""
    with np.load(checkpoint, allow_pickle=False) as arrays:
        weights = [
            (name, np.asarray(arrays[name]))
            for name in arrays.files
            if name.endswith(".W") and arrays[name].ndim == 4
        ]
    return sorted(weights, key=lambda item: _natural_key(item[0]))


def _checkpoint_weights_path(client, run):
    manifest_path = artifact_file(
        client,
        run,
        "checkpoints/checkpoint_manifest.json",
    )
    if manifest_path is None:
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        final = manifest.get("final")
        if not isinstance(final, dict) or not final.get("path"):
            return None
        final_path = Path(str(final["path"]))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None

    candidates = []
    if final_path.is_dir():
        candidates.append(final_path / "model_parameters.npz")
    else:
        candidates.append(final_path)
    if run.local_artifact_root is not None:
        candidates.extend(
            (
                run.local_artifact_root.parent.parent
                / "checkpoints"
                / run.local_artifact_root.name
                / "final.npz",
                run.local_artifact_root
                / "checkpoints"
                / final_path.name
                / "model_parameters.npz",
            )
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate, manifest

    if final_path.suffix == ".npz":
        remote_path = "checkpoints/final.npz"
    else:
        remote_path = f"checkpoints/{final_path.name}/model_parameters.npz"
    downloaded = artifact_file(client, run, remote_path)
    if downloaded is None:
        return None
    return downloaded, manifest
