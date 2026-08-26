from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dlfs.ds1.analysis.e11_cnn_filters import (
    RUN_GROUPS,
    _checkpoint_weights_path,
    _conv_weights,
    _filter_mosaic,
    _panel_output,
    _render_panel,
    _shared_weight_limit,
    _visualization_runs,
)
from repro_core.analysis.core import RunRef


class _NoDownloadClient:
    def download_artifacts(self, run_id, artifact_path):
        raise FileNotFoundError((run_id, artifact_path))


def test_visualization_runs_selects_only_seed_index_zero() -> None:
    runs = [
        RunRef("seed-1", "CNN", "1", 0, None),
        RunRef("seed-2", "CNN", "2", 0, None),
        RunRef("seed-10", "CNN", "10", 0, None),
    ]

    selected = _visualization_runs(runs)

    assert [run.seed for run in selected] == ["1"]


def test_run_groups_compare_only_the_three_simple_cnns() -> None:
    assert RUN_GROUPS == (
        ("GT06", "CNN-SIMPLE-BOOK"),
        ("GT08", "CNN-SIMPLE-SPATIAL"),
        ("GT08", "CNN-SIMPLE-SPATIAL-PERMUTED"),
    )


def test_shared_weight_limit_covers_every_panel_symmetrically() -> None:
    limit = _shared_weight_limit(
        [
            np.asarray([-0.5, 0.25]),
            np.asarray([-2.0, 1.0]),
            np.asarray([0.0, 1.5]),
        ]
    )

    assert limit == 2.0


def test_panel_output_gives_each_condition_its_own_image() -> None:
    output = Path("/tmp/e11.png")

    assert _panel_output(output, "GT06", "CNN-SIMPLE-BOOK") == Path(
        "/tmp/e11_gt06_cnn-simple-book.png"
    )


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


def test_implemented_filter_panel_has_no_titles(monkeypatch, tmp_path: Path) -> None:
    captured = []
    monkeypatch.setattr(
        "dlfs.ds1.analysis.e11_cnn_filters.save_figure",
        lambda figure, _path: captured.append(figure),
    )

    _render_panel(
        ("GT06", "CNN-SIMPLE-BOOK", np.zeros((1, 1, 3, 3))),
        output=tmp_path / "filters.png",
        limit=1.0,
    )

    assert len(captured) == 1
    assert captured[0]._suptitle is None
    assert all(axis.get_title() == "" for axis in captured[0].axes)
    image = captured[0].axes[0].images[0]
    assert image.get_clim() == (-0.9, 0.9)


def test_original_filter_panel_has_no_titles(monkeypatch, tmp_path: Path) -> None:
    from dlfs.ds1.original.native_analysis import e11

    captured = []
    monkeypatch.setattr(e11, "save", lambda _path: captured.append(e11.plt.gcf()))

    e11._filter_show(np.zeros((1, 1, 3, 3)), tmp_path / "filters.png")

    assert len(captured) == 1
    assert captured[0]._suptitle is None
    assert all(axis.get_title() == "" for axis in captured[0].axes)
    assert captured[0].axes[0].images[0].get_clim() == (-0.9, 0.9)
    e11.plt.close(captured[0])


def test_checkpoint_resolver_supports_v2_final_directory(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts" / "run-key"
    manifest_path = artifact_root / "checkpoints" / "checkpoint_manifest.json"
    checkpoint = tmp_path / "checkpoints" / "latest-generation"
    manifest_path.parent.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    np.savez(
        checkpoint / "model_parameters.npz", **{"layers.0.W": np.zeros((1, 1, 3, 3))}
    )
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

    class AnalysisInput:
        def artifact_file(self, _run, artifact_path):
            candidate = artifact_root / artifact_path
            return candidate if candidate.is_file() else None

    resolved = _checkpoint_weights_path(AnalysisInput(), run)

    assert resolved is not None
    assert resolved[0] == checkpoint / "model_parameters.npz"
