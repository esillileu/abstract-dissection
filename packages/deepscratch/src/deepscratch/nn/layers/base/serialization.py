"""Layer parameter serialization and deserialization in NPZ format."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .layer import Layer


def save_layer_params_npz(layer: Layer, path: str | Path) -> None:
    path = Path(path)
    arrays: dict[str, Any] = {}

    for name, param in layer.named_parameters():
        arrays[name] = param.backend.to_numpy(param.data).copy()

    np.savez(path, **arrays)


def load_layer_params_npz(
    layer: Layer,
    path: str | Path,
    strict: bool = True,
) -> None:
    path = Path(path)
    named_params = dict(layer.named_parameters())

    with np.load(path, allow_pickle=False) as data:
        saved_names = set(data.files)
        current_names = set(named_params)

        if strict:
            missing = current_names - saved_names
            unexpected = saved_names - current_names

            if missing:
                raise KeyError(f"missing parameters: {sorted(missing)}")

            if unexpected:
                raise KeyError(f"unexpected parameters: {sorted(unexpected)}")

        for name, param in named_params.items():
            if name not in data:
                continue

            array = data[name]

            if array.shape != param.data.shape:
                raise ValueError(
                    f"shape mismatch for {name!r}: "
                    f"expected {param.data.shape}, got {array.shape}"
                )

            new_data = param.backend.asarray(array, dtype=param.data.dtype)
            param.data[...] = new_data
