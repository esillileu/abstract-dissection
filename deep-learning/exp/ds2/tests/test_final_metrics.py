from __future__ import annotations

import json

import numpy as np

from exp.analyze import RunRef
from exp.model_parameters import ParameterCount
from exp.ds2.analyze.final_metrics import (
    FINAL_LOSS,
    FINAL_TEST_ACCURACY,
    FINAL_TEST_PERPLEXITY,
    MetricSummary,
    _format_summary,
    _write_summaries,
    summarize_atomic_runs,
)


class FakeClient:
    def download_artifacts(self, run_id, artifact_path):
        raise FileNotFoundError((run_id, artifact_path))


def _run_refs(tmp_path):
    run_refs = []
    for index, (loss, accuracy, timing_ns) in enumerate(
        (
            (0.2, 0.8, (1_000_000_000, 2_000_000_000)),
            (0.3, 0.9, (4_000_000_000,)),
        )
    ):
        root = tmp_path / f"seed-{index}"
        observations = root / "observations"
        observations.mkdir(parents=True)
        (observations / "source_curves.csv").write_text(
            "series_id,plot_index,metric,value\n"
            f"source,0,book_loss,{loss + 1}\n"
            f"source,1,book_loss,{loss}\n"
            f"source,0,exact_match_accuracy,{accuracy - 0.1}\n"
            f"source,1,exact_match_accuracy,{accuracy}\n",
            encoding="utf-8",
        )
        (root / "timing_windows.csv").write_text(
            "start_update,end_update,train_wall_time_ns\n"
            + "".join(f"1,2,{value}\n" for value in timing_ns),
            encoding="utf-8",
        )
        run_refs.append(
            RunRef(
                run_id=f"run-{index}",
                atomic_run_id="condition",
                seed=str(index),
                start_time=index,
                local_artifact_root=root,
            )
        )
    return run_refs


def test_loss_summary_uses_last_source_value_without_percent_scaling(tmp_path):
    performance, training_time = summarize_atomic_runs(
        FakeClient(),
        _run_refs(tmp_path),
        FINAL_LOSS,
    )

    assert performance is not None
    assert training_time is not None
    np.testing.assert_allclose(
        [
            performance.mean,
            performance.standard_deviation,
            performance.minimum,
            performance.maximum,
        ],
        [0.25, np.sqrt(0.005), 0.2, 0.3],
    )
    np.testing.assert_allclose(
        [
            training_time.mean,
            training_time.standard_deviation,
            training_time.minimum,
            training_time.maximum,
        ],
        [3.5, np.sqrt(0.5), 3.0, 4.0],
    )


def test_training_summary_prefers_synchronized_completed_device_time(
    tmp_path,
) -> None:
    run_refs = _run_refs(tmp_path)
    expected = (5.0, 7.0)
    for run, seconds in zip(run_refs, expected, strict=True):
        profiles = run.local_artifact_root / "profiles"
        profiles.mkdir()
        (profiles / "profiling_summary.json").write_text(
            json.dumps(
                {
                    "metrics": {
                        "runtime.train_synchronized.mean_ms": seconds * 1_000,
                    }
                }
            ),
            encoding="utf-8",
        )

    _performance, training_time = summarize_atomic_runs(
        FakeClient(),
        run_refs,
        FINAL_LOSS,
    )

    assert training_time is not None
    assert training_time.mean == 6.0
    assert training_time.run_count == 2


def test_synchronized_and_async_training_times_are_not_mixed(tmp_path) -> None:
    run_refs = _run_refs(tmp_path)
    profiles = run_refs[0].local_artifact_root / "profiles"
    profiles.mkdir()
    (profiles / "profiling_summary.json").write_text(
        json.dumps(
            {"metrics": {"runtime.train_synchronized.mean_ms": 5_000}}
        ),
        encoding="utf-8",
    )

    _performance, training_time = summarize_atomic_runs(
        FakeClient(),
        run_refs,
        FINAL_LOSS,
    )

    assert training_time is not None
    assert training_time.mean == 5.0
    assert training_time.run_count == 1


def test_accuracy_summary_is_rendered_as_fixed_two_digit_percent(tmp_path):
    performance, _training_time = summarize_atomic_runs(
        FakeClient(),
        _run_refs(tmp_path),
        FINAL_TEST_ACCURACY,
    )

    assert performance is not None
    assert _format_summary(
        "final_test_accuracy",
        performance,
        scale=100,
        decimals=2,
        unit_label=" (%)",
    ) == "final_test_accuracy (%): 85.00 ± 7.07, [80.00, 90.00], n=2"


def test_terminal_metric_falls_back_to_final_json(tmp_path):
    root = tmp_path / "run"
    metrics = root / "metrics"
    metrics.mkdir(parents=True)
    (root / "evaluations.csv").write_text(
        "axis,axis_step,split,metric,value\n",
        encoding="utf-8",
    )
    (root / "timing_windows.csv").write_text(
        "train_wall_time_ns\n1000000000\n",
        encoding="utf-8",
    )
    (metrics / "final.json").write_text(
        '{"final/test/perplexity": 134.5}',
        encoding="utf-8",
    )
    run_ref = RunRef(
        run_id="run",
        atomic_run_id="LM-LSTM",
        seed="1",
        start_time=1,
        local_artifact_root=root,
    )

    performance, _training_time = summarize_atomic_runs(
        FakeClient(),
        [run_ref],
        FINAL_TEST_PERPLEXITY,
    )

    assert performance is not None
    assert performance.mean == 134.5


def test_summary_csv_includes_model_parameter_count(tmp_path):
    output = tmp_path / "summary.csv"
    summary = MetricSummary(0.25, 0.05, 0.2, 0.3, 2)

    _write_summaries(
        output,
        {"MODEL": (FINAL_LOSS, summary, summary)},
        {"MODEL": ParameterCount(value=1234, run_count=2)},
    )

    rows = output.read_text(encoding="utf-8").splitlines()
    assert rows[-1] == "MODEL,parameter_count,2,parameters,1234,,,"
