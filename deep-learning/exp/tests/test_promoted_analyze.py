from types import SimpleNamespace

import pytest

from exp.original.promoted_analyze import _summary


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
