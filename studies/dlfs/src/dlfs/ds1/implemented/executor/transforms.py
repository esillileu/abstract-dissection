from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def _apply_input_transform(
    *, dataset: dict[str, object], backend, x_train, x_test, artifact_root: Path
) -> dict[str, object]:
    """Create deterministic, data-only transforms without consuming training RNG streams."""
    transform = dataset.get("input_transform", {"name": "identity"})
    if not isinstance(transform, dict):
        raise ValueError("dataset.input_transform must be a mapping")
    name = str(transform.get("name", "identity"))
    if name == "identity":
        return {"name": name}
    if name != "pixel_permutation":
        raise ValueError(f"unknown dataset input transform: {name}")
    if len(x_train.shape) not in {2, 4} or len(x_test.shape) != len(x_train.shape):
        raise ValueError("pixel_permutation requires flat MNIST or NCHW image tensors")
    feature_count = int(np.prod(x_train.shape[1:]))
    if feature_count != int(np.prod(x_test.shape[1:])):
        raise ValueError("train and test tensors must have the same feature count")
    seed = int(transform["seed"])
    permutation = (
        np.random.default_rng(seed).permutation(feature_count).astype(np.int64)
    )
    digest = hashlib.sha256(permutation.tobytes()).hexdigest()
    target = artifact_root / "data" / "pixel_permutation.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, permutation)
    return {
        "name": name,
        "seed": seed,
        "feature_count": feature_count,
        "sha256": digest,
        "artifact": str(target.relative_to(artifact_root)),
        "permutation": backend.xp.asarray(permutation),
    }


def _permute_pixels(x, permutation):
    original_shape = x.shape
    return x.reshape(len(x), -1)[:, permutation].reshape(original_shape)


def _validation_probe(
    *, dataset: dict[str, object], backend, x_train, t_train, artifact_root: Path
):
    size = int(dataset.get("validation_size", 0))
    if size == 0:
        return x_train, t_train, None, None, {"size": 0}
    if not 0 < size < len(x_train):
        raise ValueError("dataset.validation_size must be between 1 and train size - 1")
    seed = int(dataset.get("validation_seed", 0))
    indices = np.random.default_rng(seed).permutation(len(x_train))
    valid_indices, train_indices = indices[:size], indices[size:]
    target = artifact_root / "data" / "validation_indices.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, valid_indices)
    xp = backend.xp
    return (
        x_train[xp.asarray(train_indices)],
        t_train[xp.asarray(train_indices)],
        x_train[xp.asarray(valid_indices)],
        t_train[xp.asarray(valid_indices)],
        {
            "size": size,
            "seed": seed,
            "artifact": str(target.relative_to(artifact_root)),
            "sha256": hashlib.sha256(valid_indices.tobytes()).hexdigest(),
        },
    )
