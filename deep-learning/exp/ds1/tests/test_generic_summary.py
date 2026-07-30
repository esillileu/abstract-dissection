from __future__ import annotations

import csv

import numpy as np

from exp.analyze import RunRef
from exp.ds1.analyze import generic_summary, render


class FakeClient:
    def download_artifacts(self, run_id, artifact_path):
        raise FileNotFoundError((run_id, artifact_path))


def _run(tmp_path, seed: int, rows: str) -> RunRef:
    root = tmp_path / f"seed-{seed}"
    root.mkdir()
    (root / "updates.csv").write_text(rows, encoding="utf-8")
    return RunRef(
        run_id=f"run-{seed}",
        atomic_run_id="MLP-OPT-SGD",
        seed=str(seed),
        start_time=seed,
        local_artifact_root=root,
    )


def test_final_loss_summary_uses_each_seed_last_update(tmp_path):
    runs = [
        _run(tmp_path, 1, "update,loss\n1,1.0\n2,0.4\n"),
        _run(tmp_path, 2, "update,loss\n1,0.8\n2,0.2\n"),
    ]

    summary = generic_summary.summarize_metric(
        FakeClient(),
        runs,
        generic_summary.FINAL_LOSS,
    )

    assert summary is not None
    np.testing.assert_allclose(
        [summary.mean, summary.standard_deviation, summary.minimum, summary.maximum],
        [0.3, np.sqrt(0.02), 0.2, 0.4],
    )


def test_render_summary_writes_metric_and_training_time_rows(
    tmp_path,
    monkeypatch,
):
    run = _run(tmp_path, 1, "update,loss\n1,0.5\n2,0.25\n")
    (run.local_artifact_root / "timing_windows.csv").write_text(
        "train_wall_time_ns\n2000000000\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        generic_summary,
        "runs",
        lambda _client, _group, atomic_ids: {
            atomic_id: [run] for atomic_id in atomic_ids
        },
    )
    monkeypatch.setitem(
        generic_summary.ANALYSES,
        "test",
        generic_summary.AnalysisSpec(
            "GT",
            ("MLP-OPT-SGD",),
            (generic_summary.FINAL_LOSS,),
        ),
    )
    output = tmp_path / "summary.csv"

    paths = generic_summary.render_summary(
        FakeClient(),
        "band",
        output,
        analysis_id="test",
    )

    assert paths == [output]
    with output.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["metric"] for row in rows] == [
        "final_train_objective",
        "training_time_s",
    ]
    assert [row["mean"] for row in rows] == ["0.250", "2.0"]


def test_every_ds1_analysis_has_a_summary_renderer():
    assert set(render.SUMMARY_RENDERERS) == set(render.RENDERERS)
