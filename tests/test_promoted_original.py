from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np

from dlfs.ds1.catalog import ORIGINAL as DS1_ORIGINAL
from dlfs.original_runtime.promoted_executor import (
    _metric_name,
    _metric_rows,
    _promote_final_checkpoint,
)
from repro_core.context.paths import RuntimePaths
from repro_core.execution import RunOptions, RunOrder, RunSelection
from repro_core.execution.planning import Planner


def test_promote_final_checkpoint(tmp_path):
    output = tmp_path / "record" / "raw"
    output.mkdir(parents=True)
    np.savez(output / "checkpoint.npz", word_vectors=np.ones((3, 2)))
    root = tmp_path / "checkpoints"

    result = _promote_final_checkpoint(
        {"checkpoint": {"save_final": True}},
        SimpleNamespace(metadata={"checkpoint_root": root}),
        output,
        {
            "final/system/completed_epochs": 10.0,
            "final/system/total_updates": 20.0,
        },
    )

    assert result == root / "final.npz"
    with np.load(result, allow_pickle=False) as archive:
        assert set(archive.files) == {"word_vectors", "W_in"}
        np.testing.assert_array_equal(archive["W_in"], archive["word_vectors"])
    pointer = json.loads((root / "latest.json").read_text(encoding="utf-8"))
    assert pointer["path"] == "final.npz"

    # Disabled policy
    assert (
        _promote_final_checkpoint(
            {"checkpoint": {"save_final": False}},
            SimpleNamespace(metadata={"checkpoint_root": tmp_path / "checkpoints"}),
            tmp_path,
            {},
        )
        is None
    )


def test_promoted_original_metrics(tmp_path) -> None:
    assert _metric_name("perplexity", split="valid") == "valid/perplexity"

    (tmp_path / "metrics.csv").write_text(
        "update,epoch,split,accuracy,loss\n0,0,train,0.10,1.5\n3,1,train,0.20,1.2\n",
        encoding="utf-8",
    )
    rows, _ = _metric_rows(tmp_path)
    assert [(step, metric) for step, metric, _ in rows] == [
        (0, "train/accuracy"),
        (0, "train/loss"),
        (1, "train/accuracy"),
        (3, "train/loss"),
    ]


def test_promoted_original_planning() -> None:
    plans = Planner(DS1_ORIGINAL).build(
        RunSelection(
            experiment_ids=("e01",), atomic_run_ids=("OPT-SGD",), seed_values="2,1"
        ),
        RunOptions(
            device="cpu",
            overrides={"budget": {"max_epochs": 1}},
            order=RunOrder.SEED_FIRST,
        ),
    )
    assert [(plan.atomic_run_id, plan.seed) for plan in plans] == [
        ("OPT-SGD", 2),
        ("OPT-SGD", 1),
    ]

    # Source imports context
    from dlfs.ds1.original.run.common import source_imports

    sentinel = object()
    previous = sys.modules.get("common", sentinel)
    source = (
        RuntimePaths.from_environment().reference("dlfs1-book") / "source"
    ).resolve()
    with source_imports(source):
        __import__("common.layers")
        assert "common.layers" in sys.modules
    if previous is sentinel:
        assert "common" not in sys.modules
    else:
        assert sys.modules["common"] is previous
