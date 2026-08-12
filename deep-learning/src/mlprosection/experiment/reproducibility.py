"""Runtime reproducibility controls used by experiment executors."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
import numpy as np

from mlprosection.core.backend import Backend, BackendConfig, make_backend, set_default_backend


@dataclass(frozen=True)
class SeedStreams:
    master: int
    model_init: int
    batch_order: int
    dropout: int
    negative_sampling: int
    dataset_split: int


def seed_streams(master: int) -> SeedStreams:
    sequence = np.random.SeedSequence(master).spawn(5)
    values = [int(child.generate_state(1)[0]) for child in sequence]
    # Preserve the original four stream values; negative sampling is appended.
    return SeedStreams(master, values[0], values[1], values[2], values[4], values[3])


def configure_runtime(config: dict[str, object]) -> tuple[Backend, SeedStreams, dict[str, object]]:
    """Apply the declared numerical environment before any model is created."""
    numerics = _mapping(config, "numerics")
    streams = seed_streams(int(config["seed"]))
    random.seed(streams.master)
    np.random.seed(streams.master)
    os.environ.setdefault("PYTHONHASHSEED", str(streams.master))
    device = str(numerics.get("device", "cpu"))
    dtype = str(numerics.get("dtype", "float64"))
    backend = make_backend(BackendConfig(device=device, dtype=dtype, seed=streams.model_init))
    backend.configure_random_streams(
        {
            "model_init": streams.model_init,
            "batch_order": streams.batch_order,
            "dropout": streams.dropout,
            "negative_sampling": streams.negative_sampling,
        }
    )
    set_default_backend(backend)
    compatibility = "exact"
    if backend.name != str(numerics.get("backend", backend.name)):
        compatibility = "mismatch"
    elif not bool(numerics.get("deterministic", False)):
        compatibility = "best_effort"
    actual = {
        "backend": backend.name,
        "device": backend.device,
        "dtype": backend.dtype_name,
        "requested_backend": numerics.get("backend", "numpy"),
        "requested_device": device,
        "requested_dtype": dtype,
        "deterministic_requested": bool(numerics.get("deterministic", False)),
        "compatibility": compatibility,
        "rng_policy": "independent_persistent_component_streams_v1",
        "rng_backend": f"{backend.name}.random.RandomState",
    }
    return backend, streams, actual


def seed_batch_order(backend: Backend, streams: SeedStreams) -> None:
    """Compatibility hook; streams are initialized once in configure_runtime."""
    backend.random_stream("batch_order")


def _mapping(config: dict[str, object], name: str) -> dict[str, object]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value
