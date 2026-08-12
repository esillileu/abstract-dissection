import csv
from types import SimpleNamespace

import pytest

from exp.original.promoted_analyze import _curves, _latest_seed_runs, _summary


def _run(run_id, *, atomic, seed, start_time):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, start_time=start_time),
        data=SimpleNamespace(
            tags={"atomic_run.id": atomic},
            params={"seed/master": str(seed)},
        ),
    )


def test_latest_seed_runs_keeps_newest_attempt_per_atomic_id_and_seed():
    runs = [
        _run("old-a-1", atomic="A", seed=1, start_time=10),
        _run("b-1", atomic="B", seed=1, start_time=20),
        _run("a-2", atomic="A", seed=2, start_time=30),
        _run("new-a-1", atomic="A", seed=1, start_time=40),
    ]

    selected = _latest_seed_runs(runs)

    assert [run.info.run_id for run in selected] == [
        "new-a-1",
        "a-2",
        "b-1",
    ]


def test_latest_seed_runs_can_select_one_seed():
    runs = [
        _run("a-1", atomic="A", seed=1, start_time=10),
        _run("a-2", atomic="A", seed=2, start_time=20),
    ]

    assert [run.info.run_id for run in _latest_seed_runs(runs, seed=2)] == [
        "a-2"
    ]


def test_promoted_original_summary_prints_domain_style_summary(tmp_path, capsys):
    runs = [
        SimpleNamespace(
            data=SimpleNamespace(
                tags={"atomic_run.id": "MODEL-A"},
                metrics={
                    "final/test/accuracy": 0.8,
                    "runtime/train_total_s": 10.0,
                },
            )
        ),
        SimpleNamespace(
            data=SimpleNamespace(
                tags={"atomic_run.id": "MODEL-A"},
                metrics={
                    "final/test/accuracy": 0.9,
                    "runtime/train_total_s": 12.0,
                },
            )
        ),
    ]

    output = tmp_path / "e03_summary.csv"
    _summary(output, runs, domain="ds1_original", experiment_id="e03")

    text = capsys.readouterr().out
    assert "ds1_original/e03 summary" in text
    assert "[MODEL-A]" in text
    assert "final_test_accuracy (%): 85.00 ± 7.07, [80.00, 90.00], n=2" in text
    assert "training_time (s): 11.0 ± 1.4, [10.0, 12.0], n=2" in text
    assert "standard_deviation" in output.read_text(encoding="utf-8")


@pytest.mark.parametrize("domain", ["ds1_original", "ds2_original"])
def test_promoted_original_summary_handles_empty_runs(tmp_path, capsys, domain):
    output = tmp_path / "summary.csv"
    _summary(output, [], domain=domain, experiment_id="e01")

    assert f"{domain}/e01 summary" in capsys.readouterr().out
    assert output.read_text(encoding="utf-8").splitlines() == [
        "atomic_run_id,metric,count,mean,standard_deviation,minimum,maximum"
    ]


class _ArtifactClient:
    def __init__(self, paths):
        self.paths = paths

    def download_artifacts(self, run_id, artifact_path):
        assert artifact_path == "raw/metrics.csv"
        return self.paths[run_id]


def _cnn_run(run_id, *, train, test, test_full, root):
    path = root / f"{run_id}.csv"
    path.write_text(
        "epoch,split,accuracy\n"
        f"0,train,0.1\n0,test,0.2\n"
        f"19,train,{train}\n19,test,{test}\n20,test-full,{test_full}\n",
        encoding="utf-8",
    )
    run = SimpleNamespace(
        info=SimpleNamespace(run_id=run_id),
        data=SimpleNamespace(
            tags={"atomic_run.id": "SIMPLE-CONVNET"},
            metrics={
                "final/test/accuracy": test_full,
                "runtime/train_total_s": 10.0,
            },
        ),
    )
    return run, path


def test_ds1_original_cnn_summary_recovers_splits_from_raw_metrics(
    tmp_path,
):
    run_1, path_1 = _cnn_run(
        "run-1", train=0.98, test=0.96, test_full=0.95, root=tmp_path
    )
    run_2, path_2 = _cnn_run(
        "run-2", train=1.0, test=0.98, test_full=0.97, root=tmp_path
    )
    client = _ArtifactClient({"run-1": path_1, "run-2": path_2})
    output = tmp_path / "e06_summary.csv"

    _summary(
        output,
        [run_1, run_2],
        domain="ds1_original",
        experiment_id="e06",
        client=client,
    )

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert [row["metric"] for row in rows] == [
        "final/test-full/accuracy",
        "final/test/accuracy",
        "final/train/accuracy",
        "runtime/train_total_s",
    ]
    train = next(row for row in rows if row["metric"] == "final/train/accuracy")
    assert float(train["mean"]) == pytest.approx(0.99)


def test_ds1_original_cnn_curve_contains_train_and_test_series(tmp_path):
    run, raw_path = _cnn_run(
        "run-1", train=0.98, test=0.96, test_full=0.95, root=tmp_path
    )
    client = _ArtifactClient({"run-1": raw_path})
    output = tmp_path / "e06_band.png"

    _curves(
        client,
        output,
        [run],
        "band",
        domain="ds1_original",
        experiment_id="e06",
    )

    rows = list(csv.DictReader(output.with_suffix(".csv").open(encoding="utf-8")))
    assert [row["series"] for row in rows] == [
        "SIMPLE-CONVNET/train",
        "SIMPLE-CONVNET/test",
    ]
