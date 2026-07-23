from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from exp.analyze import RunRef
from exp.ds1.analyze.e11_cnn_filters import (
    _checkpoint_weights_path,
    _conv_weights,
    _filter_mosaic,
)


class _NoDownloadClient:
    def download_artifacts(self, run_id, artifact_path):
        raise FileNotFoundError((run_id, artifact_path))


def test_conv_weights_selects_only_four_dimensional_weights_in_model_order(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "final.npz"
    np.savez(
        checkpoint,
        **{
            "layers.10.W": np.zeros((4, 2, 3, 3)),
            "layers.2.W": np.zeros((2, 1, 3, 3)),
            "layers.2.b": np.zeros(2),
            "layers.12.W": np.zeros((36, 5)),
        },
    )

    layers = _conv_weights(checkpoint)

    assert [name for name, _weights in layers] == ["layers.2.W", "layers.10.W"]


def test_filter_mosaic_retains_every_weight() -> None:
    weights = np.arange(2 * 3 * 2 * 2, dtype=float).reshape(2, 3, 2, 2)

    mosaic = _filter_mosaic(weights)

    finite = mosaic[np.isfinite(mosaic)]
    np.testing.assert_array_equal(np.sort(finite), np.arange(weights.size))


def test_checkpoint_resolver_supports_v2_final_directory(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts" / "run-key"
    manifest_path = artifact_root / "checkpoints" / "checkpoint_manifest.json"
    checkpoint = tmp_path / "checkpoints" / "latest-generation"
    manifest_path.parent.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    np.savez(checkpoint / "model_parameters.npz", **{"layers.0.W": np.zeros((1, 1, 3, 3))})
    manifest_path.write_text(
        json.dumps(
            {
                "format": "v2",
                "final": {
                    "path": str(checkpoint),
                    "epoch": 2,
                    "update": 20,
                },
            }
        ),
        encoding="utf-8",
    )
    run = RunRef(
        run_id="run",
        atomic_run_id="CNN",
        seed="1",
        start_time=0,
        local_artifact_root=artifact_root,
    )

    resolved = _checkpoint_weights_path(_NoDownloadClient(), run)

    assert resolved is not None
    assert resolved[0] == checkpoint / "model_parameters.npz"
