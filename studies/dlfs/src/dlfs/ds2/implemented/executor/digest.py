from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def _sequence_dataset_path(file_name: str) -> Path:
    from repro_core.context.paths import RuntimePaths

    return RuntimePaths.from_environment().dataset("sequence") / file_name


def _array_digest(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.asarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _path_digest(path: Path) -> str:
    if path.is_file():
        return _file_digest(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(_file_digest(child).encode())
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
