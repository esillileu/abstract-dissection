from types import SimpleNamespace

import pytest

from exp.original.promoted_analyze import _latest_seed_runs, _summary


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
