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
from exp.original.runtime_context import array_module, device


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
    gpu = gpu and device().startswith("cuda")
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
        if gpu:
            config = importlib.import_module("common.config")
            config.GPU = True
            # CuPy 14 makes ``cp.add.at`` read-only, while the snapshot assigns
            # ``cp.scatter_add`` to it.  Exporting the same CuPy module through
            # the book's compatibility module preserves the scatter-add
            # semantics without editing the immutable source checkout.
            cp = array_module()
            compatibility = types.ModuleType("common.np")
            compatibility.GPU = True
            compatibility.np = cp
            sys.modules["common.np"] = compatibility
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
    return name.split(".", 1)[0] in {
        "b2",
        "common",
        "dataset",
        "seq2seq",
        "ch03",
        "ch04",
        "ch05",
        "ch06",
        "ch07",
        "ch08",
    }


def install_b2_compatibility_aliases() -> None:
    """Map the migrated ch05 ``b2.common`` imports back to book modules."""
    b2 = types.ModuleType("b2")
    b2.__path__ = []  # type: ignore[attr-defined]
    b2_common = types.ModuleType("b2.common")
    b2_common.__path__ = []  # type: ignore[attr-defined]
    sys.modules["b2"] = b2
    sys.modules["b2.common"] = b2_common
    for name in ("np", "time_layers"):
        sys.modules[f"b2.common.{name}"] = importlib.import_module(f"common.{name}")


def install_ch07_compatibility_aliases() -> None:
    """Support ch07's script-relative ``from seq2seq`` import."""
    sys.modules["seq2seq"] = importlib.import_module("ch07.seq2seq")


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


def to_device(util, value):
    """Use the selected backend without invoking upstream's hard-coded CuPy helper."""
    if device().startswith("cuda"):
        return util.to_gpu(value)
    return np.asarray(value)


def evaluate_seq2seq(model, x_test, t_test, *, reverse: bool):
    correct_count = 0
    predictions = []
    for index in range(len(x_test)):
        question = x_test[[index]]
        correct = t_test[[index]]
        start_id = int(correct.flatten()[0])
        generated = model.generate(question, start_id, len(correct.flatten()) - 1)
        target = to_host(correct.flatten()[1:])
        generated_host = to_host(generated)
        matched = bool(np.array_equal(generated_host, target))
        correct_count += int(matched)
        if index < 10:
            predictions.append(
                {
                    "example_id": index,
                    "source_ids": " ".join(map(str, to_host(question).flatten())),
                    "target_ids": " ".join(map(str, target)),
                    "prediction_ids": " ".join(map(str, generated_host)),
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
    "install_b2_compatibility_aliases",
    "install_ch07_compatibility_aliases",
    "load_npz",
    "np",
    "restore_params",
    "save_csv",
    "save_npz",
    "source_imports",
    "to_device",
    "to_host",
]
