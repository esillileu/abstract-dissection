"""Shared model parameter and RNG state checkpoint serialization adapter."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


class CheckpointAdapter:
    """Serializes and restores model weights, optimizer states, and RNG buffers."""

    @staticmethod
    def save(state: dict[str, Any], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @staticmethod
    def load(path: Path) -> dict[str, Any]:
        with path.open("rb") as f:
            return pickle.load(f)


__all__ = ["CheckpointAdapter"]
