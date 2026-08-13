from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np

from exp.analyze import RunRef
from exp.deepscratch.ds1.analysis import e12_summary, final_metrics


class FakeClient:
    def download_artifacts(self, run_id, artifact_path):
        raise FileNotFoundError((run_id, artifact_path))

    def get_run(self, run_id):
        index = int(run_id.rsplit("-", 1)[1])
        train_accuracy = (0.98, 0.99)[index]
        return SimpleNamespace(
            data=SimpleNamespace(
                metrics={
                    "final/train-full/accuracy": train_accuracy,
                    "final/train-test/accuracy-gap": train_accuracy - 0.999,
                }
            )
        )


def test_e12_model_order_includes_simple_convnet():
    assert e12_summary.MODELS == (
        ("GT09", "MLP-EXT-ALL-BOOK"),
        ("GT06", "CNN-SIMPLE-BOOK"),
        ("GT07", "CNN-DEEP-BOOK"),
    )


def _run_refs(tmp_path):
    run_refs = []
    for index, (train_accuracy, test_accuracy, timing_ns) in enumerate(
        (
            (0.98, 0.96, (1_000_000_000, 2_000_000_000)),
            (0.99, 0.97, (4_000_000_000,)),
        )
    ):
        root = tmp_path / f"seed-{index}"
        root.mkdir()
        (root / "evaluations.csv").write_text(
            "axis,axis_step,update,epoch,evaluation_set_id,split,"
            "example_count,loss,accuracy\n"
            "update,1,1,1,mnist-train-first-1000,train,1000,,0.50\n"
            "update,1,1,1,mnist-test-first-1000,test,1000,,0.40\n"
            f"update,20,20,20,mnist-train-first-1000,train,1000,,"
            f"{train_accuracy}\n"
            f"update,20,20,20,mnist-test-first-1000,test,1000,,"
            f"{test_accuracy}\n"
            "terminal,12000,12000,20,mnist-test-full,test,10000,,0.999\n",
            encoding="utf-8",
        )
        (root / "timing_windows.csv").write_text(
            "train_wall_time_ns\n"
            + "".join(f"{value}\n" for value in timing_ns),
            encoding="utf-8",
        )
        run_refs.append(
            RunRef(
                run_id=f"run-{index}",
                atomic_run_id="CNN-SIMPLE-BOOK",
                seed=str(index),
                start_time=index,
                local_artifact_root=root,
            )
        )
    return run_refs


def test_accuracy_summary_uses_sampled_train_and_full_test_values(tmp_path):
    summaries = final_metrics.accuracy_summaries_for_runs(
        FakeClient(),
        {"CNN-SIMPLE-BOOK": _run_refs(tmp_path)},
    )

    train = summaries["CNN-SIMPLE-BOOK/train_accuracy"]
    test = summaries["CNN-SIMPLE-BOOK/test_accuracy"]
    training_time = summaries["CNN-SIMPLE-BOOK/training_time_s"]
    assert train is not None
    assert test is not None
    assert training_time is not None
    np.testing.assert_allclose(
        [train.mean, train.standard_deviation],
        [0.985, np.sqrt(0.00005)],
    )
    np.testing.assert_allclose(
        [test.mean, test.standard_deviation],
        [0.999, 0.0],
    )
    np.testing.assert_allclose(
        [training_time.mean, training_time.standard_deviation],
        [3.5, np.sqrt(0.5)],
    )


def test_accuracy_summary_prints_compact_format_and_writes_original_columns(
    tmp_path,
    monkeypatch,
    capsys,
):
    original_root = tmp_path / "original"
    trial_root = original_root / "e06" / "dlfs1.ch07.simple-convnet"
    trial_root.mkdir(parents=True)
    (trial_root / "metrics.csv").write_text(
        "epoch,split,accuracy\n"
        "0,train,0.20\n"
        "0,test,0.19\n"
        "19,train,0.998\n"
        "19,test,0.979\n"
        "20,test-full,0.9882\n",
        encoding="utf-8",
    )
    (original_root.parent / "cupy_estimate.json").write_text(
        '{"results": ['
        '{"experiment_id": "e06", "projected_update_time_s": 12.34}'
        "]}\n",
        encoding="utf-8",
    )
    run_refs = _run_refs(tmp_path)
    monkeypatch.setattr(
        final_metrics,
        "runs",
        lambda _client, group_id, atomic_run_ids: {
            atomic_run_ids[0]: run_refs
        },
    )
    output = tmp_path / "e06_summary.csv"

    paths = final_metrics.render_accuracy_comparison_summary(
        FakeClient(),
        analysis_id="e06 summary",
        group_id="GT06",
        atomic_run_ids=["CNN-SIMPLE-BOOK"],
        output=output,
        original_data_root=original_root,
    )

    printed = capsys.readouterr().out
    assert "train_accuracy (%): 98.50 ± 0.71 (n=2)" in printed
    assert "test_accuracy (%): 99.90 ± 0.00 (n=2)" in printed
    assert "training_time (s): 3.5 ± 0.7 (n=2)" in printed
    assert "original" not in printed
    assert paths == [output]
    with output.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["metric"] for row in rows] == [
        "train_accuracy",
        "test_accuracy",
        "training_time_s",
    ]
    assert [row["evaluation_set"] for row in rows] == [
        "mnist-train-first-1000",
        "mnist-test-full",
        "",
    ]
    assert [row["original"] for row in rows] == ["99.80", "98.82", "12.3"]
    assert [row["original_kind"] for row in rows] == [
        "measured",
        "measured",
        "projected",
    ]
    assert [row["mean"] for row in rows] == ["98.50", "99.90", "3.5"]
    assert [row["standard_deviation"] for row in rows] == [
        "0.71",
        "0.00",
        "0.7",
    ]


def test_cross_group_summary_prints_full_accuracies_time_and_gap(
    tmp_path,
    monkeypatch,
    capsys,
):
    run_refs = _run_refs(tmp_path)
    monkeypatch.setattr(
        final_metrics,
        "runs",
        lambda _client, group_id, atomic_run_ids: {
            atomic_run_ids[0]: run_refs
        },
    )
    output = tmp_path / "e12_summary.csv"

    final_metrics.render_cross_group_summary(
        FakeClient(),
        analysis_id="e12 summary",
        models=(("GT07", "CNN-DEEP-BOOK"),),
        output=output,
    )

    printed = capsys.readouterr().out
    assert "train_accuracy (%): 98.50 ± 0.71 (n=2)" in printed
    assert "test_accuracy (%): 99.90 ± 0.00 (n=2)" in printed
    assert "training_time (s): 3.5 ± 0.7 (n=2)" in printed
    assert "train_test_gap (%): -1.40 ± 0.71 (n=2)" in printed
    with output.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["metric"] for row in rows] == [
        "train_accuracy",
        "test_accuracy",
        "training_time_s",
        "train_test_gap",
    ]
    assert rows[0]["evaluation_set"] == "mnist-train-full"
    assert rows[3]["evaluation_set"] == (
        "mnist-train-full - mnist-test-full"
    )
