"""Shared adapter helpers for DS1 book-one source modules."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from exp.deepscratch.original_runtime.cache_protocol import save_csv, save_npz, to_host
from exp.deepscratch.original_runtime.runtime_context import array_module


Runner = Callable[[Path, Path], None]


@dataclass(frozen=True)
class Trial:
    trial_id: str
    backend: str
    conditions: dict[str, object]
    source_files: tuple[str, ...]
    runner: Runner


COMMON_SOURCES = (
    "common/functions.py",
    "common/gradient.py",
    "common/layers.py",
    "common/multi_layer_net.py",
    "common/multi_layer_net_extend.py",
    "common/optimizer.py",
    "common/trainer.py",
    "common/util.py",
    "dataset/mnist.py",
)


@contextmanager
def source_imports(worktree: Path) -> Iterator[None]:
    value = str(worktree)
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, value)
    old_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if _is_upstream_name(name)
    }
    for name in old_modules:
        del sys.modules[name]
    try:
        yield
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
        for name in tuple(sys.modules):
            if _is_upstream_name(name):
                del sys.modules[name]
        sys.modules.update(old_modules)
        if value in sys.path:
            sys.path.remove(value)


def _is_upstream_name(name: str) -> bool:
    root = name.split(".", 1)[0]
    return root in {"common", "dataset", "ch04", "ch05", "ch06", "ch07", "ch08"}


def load_mnist(worktree: Path, *, flatten: bool = True):
    with source_imports(worktree):
        loader = importlib.import_module("dataset.mnist").load_mnist
        return loader(flatten=flatten)


def patch_cupy_modules(worktree: Path, names: tuple[str, ...]):
    """Import book modules and replace only their module-global ``np`` names."""
    cp = array_module()
    modules = {name: importlib.import_module(name) for name in names}
    for module in modules.values():
        if hasattr(module, "np"):
            module.np = cp
    return cp, modules


def save_params(path: Path, params: dict[str, object], **extra: object) -> None:
    arrays = {f"param__{key}": to_host(value) for key, value in params.items()}
    arrays.update(extra)
    save_npz(path, **arrays)


def rows_for_series(
    values: list[object],
    *,
    condition: str,
    metric: str,
    batch_size: int,
) -> list[dict[str, object]]:
    return [
        {
            "update": index + 1,
            "epoch": "",
            "condition": condition,
            "metric": metric,
            "value": float(to_host(value)),
            "batch_size": batch_size,
        }
        for index, value in enumerate(values)
    ]


__all__ = [
    "COMMON_SOURCES",
    "Trial",
    "importlib",
    "np",
    "patch_cupy_modules",
    "rows_for_series",
    "save_csv",
    "save_npz",
    "save_params",
    "source_imports",
    "to_host",
]
