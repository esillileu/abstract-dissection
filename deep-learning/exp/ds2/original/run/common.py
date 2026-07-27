"""Shared adapters for the second book's source modules."""

from __future__ import annotations

import importlib
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from exp.original.cache import load_npz, save_csv, save_npz, to_host


Runner = Callable[[Path, Path, Path], None]


@dataclass(frozen=True)
class Trial:
    trial_id: str
    backend: str
    conditions: dict[str, object]
    source_files: tuple[str, ...]
    runner: Runner


COMMON_SOURCES = (
    "common/base_model.py",
    "common/config.py",
    "common/functions.py",
    "common/layers.py",
    "common/np.py",
    "common/optimizer.py",
    "common/time_layers.py",
    "common/trainer.py",
    "common/util.py",
    "dataset/ptb.py",
    "dataset/sequence.py",
)


@contextmanager
def source_imports(worktree: Path, *, gpu: bool = False) -> Iterator[None]:
    value = str(worktree)
    sys.path.insert(0, value)
    old_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if _is_upstream_name(name)
    }
    for name in old_modules:
        del sys.modules[name]
    try:
        if gpu:
            config = importlib.import_module("common.config")
            config.GPU = True
            # CuPy 14 makes ``cp.add.at`` read-only, while the snapshot assigns
            # ``cp.scatter_add`` to it.  Exporting the same CuPy module through
            # the book's compatibility module preserves the scatter-add
            # semantics without editing the immutable source checkout.
            cp = importlib.import_module("cupy")
            compatibility = types.ModuleType("common.np")
            compatibility.GPU = True
            compatibility.np = cp
            sys.modules["common.np"] = compatibility
        yield
    finally:
        for name in tuple(sys.modules):
            if _is_upstream_name(name):
                del sys.modules[name]
        sys.modules.update(old_modules)
        if value in sys.path:
            sys.path.remove(value)


def _is_upstream_name(name: str) -> bool:
    return name.split(".", 1)[0] in {
        "common",
        "dataset",
        "ch03",
        "ch04",
        "ch05",
        "ch06",
        "ch07",
        "ch08",
    }


def checkpoint_arrays(params: list[object]) -> dict[str, np.ndarray]:
    return {
        f"param_{index:03d}": to_host(param)
        for index, param in enumerate(params)
    }


def restore_params(params: list[object], archive: dict[str, np.ndarray]) -> None:
    for index, param in enumerate(params):
        value = archive[f"param_{index:03d}"]
        if hasattr(param, "set"):
            param.set(value)
        else:
            param[...] = value


def evaluate_seq2seq(model, x_test, t_test, *, reverse: bool):
    correct_count = 0
    predictions = []
    for index in range(len(x_test)):
        question = x_test[[index]]
        correct = t_test[[index]]
        start_id = correct.flatten()[0]
        generated = model.generate(question, start_id, len(correct.flatten()) - 1)
        target = correct.flatten()[1:]
        matched = bool(np.array_equal(np.asarray(generated), np.asarray(target)))
        correct_count += int(matched)
        if index < 10:
            predictions.append(
                {
                    "example_id": index,
                    "source_ids": " ".join(map(str, question.flatten())),
                    "target_ids": " ".join(map(str, target)),
                    "prediction_ids": " ".join(map(str, generated)),
                    "exact_match": matched,
                    "reverse": reverse,
                }
            )
    return correct_count / len(x_test), predictions


__all__ = [
    "COMMON_SOURCES",
    "Trial",
    "checkpoint_arrays",
    "evaluate_seq2seq",
    "importlib",
    "load_npz",
    "np",
    "restore_params",
    "save_csv",
    "save_npz",
    "source_imports",
    "to_host",
]
