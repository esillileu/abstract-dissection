from __future__ import annotations

import csv

from mlprosection import Tensor
from mlprosection.events import EpochEvent, EvaluationResult, SourceObjectiveSample, TrainEndEvent, TrainingWindowEvent, UpdateEvent
from mlprosection.experiment.event_executor import EvaluationRequest, EventExperimentExecutor
from mlprosection.experiment.executor import ExperimentContext

from exp.ds2.executor import Word2VecExecutor, _source_curve_from_objective
from exp.ds2.records import DS2Records
from exp.ds2.spec import parse_run_spec


def test_ds2_records_long_metrics_and_source_samples() -> None:
    request = EvaluationRequest("ptb-valid", "valid", object(), ("perplexity",))
    executor = EventExperimentExecutor(
        records=DS2Records(),
        evaluate=lambda _request: EvaluationResult(
            example_count=12, loss=1.2, accuracy=None, unit="token", unit_count=12,
            perplexity=3.32,
        ),
        epoch_requests=lambda _event: (request,),
        source_curve=lambda event: {
            "series_id": "book-ppl", "plot_index": event.local_iteration,
            "update_end": event.update, "value": event.objective,
        },
    )
    executor.begin()
    executor.on_source_objective(SourceObjectiveSample(1, 1, 0, Tensor(1.5), 20))
    executor.on_update(UpdateEvent(1, 1, 2, Tensor(1.4), 0.1))
    executor.on_epoch(EpochEvent(1, 1, 1, 2))
    executor.on_train_end(TrainEndEvent("completed", 1, 1))

    assert executor.records.source_samples[0]["local_iteration"] == 0
    assert executor.records.source_curves[0]["series_id"] == "book-ppl"
    assert {row["metric"] for row in executor.records.evaluations} == {"loss", "perplexity"}


def test_ds2_mlflow_metric_rows_include_device_timing_when_present() -> None:
    records = DS2Records()
    records.add_timing_window(TrainingWindowEvent(
        start_update=1,
        end_update=3,
        update_count=3,
        closed_by="epoch_end",
        train_wall_time_ns=2_000_000,
        eval_wall_time_ns=1_000_000,
        train_device_time_ns=750_000,
        eval_device_time_ns=250_000,
    ))

    rows = records.mlflow_metric_rows()

    assert (3, "update/runtime/window/train_device_time_ms", 0.75) in rows
    assert (3, "update/runtime/window/eval_device_time_ms", 0.25) in rows


def test_ds2_source_curve_from_objective_uses_documented_schema() -> None:
    reducer = _source_curve_from_objective({
        "recording": {
            "source_curve": {
                "kind": "interval_mean_loss",
                "every_updates": 2,
                "reducer": "mean",
                "plot_index": "zero_based_append",
            }
        }
    })

    assert reducer(SourceObjectiveSample(1, 1, 1, Tensor(1.0), 3)) is None
    point = reducer(SourceObjectiveSample(2, 1, 2, Tensor(3.0), 5))

    assert point == {
        "series_id": "interval_mean_loss",
        "plot_index": 1,
        "update_start": 1,
        "update_end": 2,
        "epoch_start": 1,
        "epoch_end": 1,
        "unit": "example",
        "unit_count": 8,
        "metric": "loss",
        "reducer": "mean",
        "value": 2.0,
    }


def test_ds2_source_curves_csv_schema_and_mlflow_mapping(tmp_path) -> None:
    records = DS2Records()
    records.add_source_curve({
        "series_id": "interval_mean_loss",
        "plot_index": 4,
        "update_start": 1,
        "update_end": 5,
        "epoch_start": 1,
        "epoch_end": 1,
        "unit": "example",
        "unit_count": 20,
        "metric": "loss",
        "reducer": "mean",
        "value": 1.5,
    })
    records.add_source_curve({
        "series_id": "full_test_exact_match",
        "plot_index": 0,
        "update_start": 10,
        "update_end": 10,
        "epoch_start": 1,
        "epoch_end": 1,
        "unit": "sequence",
        "unit_count": 1000,
        "metric": "exact_match_accuracy",
        "reducer": "identity",
        "value": 0.42,
    })

    records.write_csv(tmp_path)

    path = tmp_path / "observations" / "source_curves.csv"
    rows = list(csv.DictReader(path.open()))
    assert list(rows[0]) == [
        "series_id", "plot_index", "update_start", "update_end",
        "epoch_start", "epoch_end", "unit", "unit_count",
        "metric", "reducer", "value",
    ]
    assert rows[0]["update_start"] == "1"
    assert rows[1]["metric"] == "exact_match_accuracy"
    assert (4, "series/train/loss", 1.5) in records.mlflow_metric_rows()
    assert (0, "series/eval_test/exact_match_accuracy", 0.42) in records.mlflow_metric_rows()


def test_ds2_executor_returns_runtime_profiling_summary(tmp_path) -> None:
    spec = parse_run_spec(
        "exp/ds2/config/e01_toy_word2vec.yaml",
        atomic_run_id="W2V-TOY-CBOW-FULL",
        overrides={"budget": {"max_epochs": 1}, "tracking": {"enabled": False}},
    )
    config = spec.config
    config["seed"] = 1

    result = Word2VecExecutor().run(
        config,
        ExperimentContext(metadata={"artifact_root": tmp_path, "checkpoint_root": tmp_path / "checkpoints"}),
    )

    assert "runtime.train_total.count" in result.profiling_metrics
    assert "memory.run.start.cpu.rss_bytes" in result.profiling_metrics
    assert "memory.run.end.cpu.rss_bytes" in result.profiling_metrics
