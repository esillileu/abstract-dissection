from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from exp.deepscratch.analysis.declarations import MetricDeclaration
from exp.deepscratch.analysis.input import AnalysisRun
from exp.deepscratch.analysis.summary import write_study_summary
from exp.deepscratch.identity import Variant, Volume
from exp.deepscratch.ds2.result_schema import SUMMARY_METRICS as DS2_SUMMARY_METRICS
from exp.framework.results import NativeRunResult


def test_ds2_e05_declares_train_validation_and_test_perplexity() -> None:
    assert [metric.metric_id for metric in DS2_SUMMARY_METRICS["e05"]] == [
        "train_perplexity",
        "validation_perplexity",
        "test_perplexity",
    ]


@pytest.mark.parametrize("study_id", ["e06", "e07"])
def test_ds2_seq2seq_summaries_declare_test_accuracy_only(study_id: str) -> None:
    metric = DS2_SUMMARY_METRICS[study_id][0]
    assert metric.metric_id == "test_accuracy"
    assert metric.unit == "percent"
    assert metric.value_scale == 100.0


def test_summary_reports_seed_statistics_time_and_parameter_count(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    native = NativeRunResult("unused", "test", 1, "book-source-v1", ())
    runs = [
        AnalysisRun(
            f"run-{seed}",
            "optimizer.sgd",
            "MLP-OPT-SGD",
            str(seed),
            Variant.IMPLEMENTED,
            native,
        )
        for seed in (1, 2)
    ]
    manifests = {}
    for run in runs:
        path = tmp_path / f"{run.run_id}.json"
        path.write_text(
            json.dumps([{"name": "weight", "numel": 12}]),
            encoding="utf-8",
        )
        manifests[run.run_id] = path

    class Input:
        variant = Variant.IMPLEMENTED
        declaration = SimpleNamespace(
            conditions=(SimpleNamespace(canonical_id="optimizer.sgd"),)
        )

        def runs(self, _condition_ids):
            return {"optimizer.sgd": runs}

        def metric_value(self, run, metric_id):
            values = {
                ("run-1", "final/train/loss"): 1.0,
                ("run-2", "final/train/loss"): 3.0,
                ("run-1", "runtime/train_total_s"): 10.0,
                ("run-2", "runtime/train_total_s"): 14.0,
            }
            return values.get((run.run_id, metric_id))

        def artifact_file(self, run, _artifact_path):
            return manifests[run.run_id]

    metrics = (
        MetricDeclaration(
            "train_loss",
            "nats",
            "train",
            "run",
            ("final/train/loss",),
            ("final/train/loss",),
        ),
        MetricDeclaration(
            "training_time_s",
            "seconds",
            "train",
            "run",
            ("runtime/train_total_s",),
            ("runtime/train_total_s",),
        ),
    )

    path = write_study_summary(
        Input(),
        volume=Volume.DS1,
        study_id="e01",
        metrics=metrics,
        output_dir=tmp_path,
        output_variants=(Variant.IMPLEMENTED,),
        print_console=True,
    )

    summary = path.read_text(encoding="utf-8")
    assert path.suffix == ".md"
    assert "## e01 / optimizer.sgd / implemented" in summary
    assert (
        "- train_loss (nats): 2.00 ± 1.41 "
        "(sample standard deviation; variance=2.00; n=2)"
    ) in summary
    assert summary.index("## e01 / optimizer.sgd / implemented") < summary.index(
        "## Detailed statistics"
    )
    assert "| optimizer.sgd | implemented | train_loss | nats | 2.00 | 1.41 | 2.00" in summary
    assert "| optimizer.sgd | implemented | training_time_s | seconds | 12.00 | 2.83 | 8.00" in summary
    assert "| optimizer.sgd | implemented | parameter_count | parameters | 12 | 0 | 0" in summary
    output = capsys.readouterr().out
    assert "[e01/optimizer.sgd/implemented]" in output
    assert "sample standard deviation; variance=" in output
