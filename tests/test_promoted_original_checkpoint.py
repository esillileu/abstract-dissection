from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from dlfs.original_runtime.promoted_executor import (
    _promote_final_checkpoint,
)


def test_promote_final_checkpoint_copies_original_archive(tmp_path):
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
    assert pointer["epoch"] == 10
    assert pointer["update"] == 20


def test_promote_final_checkpoint_respects_disabled_policy(tmp_path):
    assert (
        _promote_final_checkpoint(
            {"checkpoint": {"save_final": False}},
            SimpleNamespace(metadata={"checkpoint_root": tmp_path / "checkpoints"}),
            tmp_path,
            {},
        )
        is None
    )
