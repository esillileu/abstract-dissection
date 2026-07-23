from __future__ import annotations

import csv
from pathlib import Path

from mlprosection import Tensor
from mlprosection.events import UpdateEvent
from mlprosection.experiment import ExperimentContext
from exp.ds1.executor import SupervisedClassificationExecutor
from exp.ds1.records import DS1Records
from exp.ds1.spec import parse_run_spec


def test_supervised_executor_consumes_forward_events_for_updates_evaluation_and_timing(
    monkeypatch, tmp_path
) -> None:
    train_x = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], backend="cpu")
    train_t = Tensor([0, 1, 0, 1], backend="cpu")
    test_x = Tensor([[1.0, 0.0], [0.0, 1.0]], backend="cpu")
    test_t = Tensor([0, 1], backend="cpu")
    monkeypatch.setattr(
        "exp.ds1.executor.load_mnist",
        lambda *, flatten, gpu: ((train_x, train_t), (test_x, test_t)),
    )

    config = {
        "kind": "supervised_classification",
        "seed": 1,
        "dataset": {"flatten": True},
        "model": {
            "alias": "MLP", "input_size": 2, "hidden_sizes": [3],
            "output_size": 2, "activation": "relu", "initializer": "he",
        },
        "optimizer": {"name": "sgd", "learning_rate": 0.01},
        "training": {
            "max_epochs": 1, "max_updates": 2,
        },
        "evaluation": {
            "sources": [
                {"id": "mnist-train-first-2", "split": "train", "kind": "first_n", "count": 2},
                {"id": "mnist-test-full", "split": "test", "kind": "full"},
            ],
            "schedule": {"on_update": {"start": 1, "every": 1, "sets": ["mnist-train-first-2", "mnist-test-full"]}},
        },
        "loader": {"batch_size": 2, "drop_last": False, "sampling_method": "permutation_per_epoch"},
        "checkpoint": {"save_on_eval": False},
        "profiling": {},
        "numerics": {"backend": "numpy", "device": "cpu", "dtype": "float64"},
    }
    context = ExperimentContext(metadata={"artifact_root": tmp_path, "checkpoint_root": tmp_path / "checkpoints"})

    result = SupervisedClassificationExecutor().run(config, context)

    update_rows = list(csv.DictReader((tmp_path / "updates.csv").open()))
    evaluation_rows = list(csv.DictReader((tmp_path / "evaluations.csv").open()))
    metric_names = {row[1] for row in result.metric_rows}
    assert [int(row["update"]) for row in update_rows] == [1, 2]
    assert [int(row["axis_step"]) for row in evaluation_rows if row["split"] == "train"] == [1, 2]
    assert "update/train/loss" in metric_names
    assert "update/eval_train/loss" in metric_names
    assert result.metrics["final/system/total_updates"] == 2.0
    assert "runtime.train_total.count" in result.profiling_metrics
    assert "memory.run.start.cpu.rss_bytes" in result.profiling_metrics
    assert "memory.run.end.cpu.rss_bytes" in result.profiling_metrics
    checkpoint_rows = list(csv.DictReader((tmp_path / "checkpoints.csv").open()))
    assert [row["kind"] for row in checkpoint_rows] == ["latest"]
    assert Path(checkpoint_rows[0]["path"]).is_dir()


def test_ds1_run_spec_rejects_legacy_trainer_policy(tmp_path) -> None:
    path = tmp_path / "legacy.yaml"
    path.write_text(
        """
kind: supervised_classification
domain: ds1
run: {experiment_id: e99, group_id: GT99, protocol: test, recipe_id: test, structure_signature: test}
training: {record_step_validation_interval: 1}
budget: {max_epochs: 1}
recording: {evaluation_sources: [], triggers: []}
variants:
  RUN: {}
""",
        encoding="utf-8",
    )

    try:
        parse_run_spec(path, atomic_run_id="RUN")
    except ValueError as exc:
        assert "training must be replaced by RunSpec budget/recording fields" in str(exc)
    else:
        raise AssertionError("expected legacy policy validation")


def test_ds1_records_flush_every_256_rows(tmp_path) -> None:
    records = DS1Records()
    records.bind_artifact_root(tmp_path)

    for update in range(1, 257):
        records.on_update(UpdateEvent(update, 1, 2, Tensor(0.1), 0.01))

    update_rows = list(csv.DictReader((tmp_path / "updates.csv").open()))
    assert len(update_rows) == 256
    assert update_rows[-1]["update"] == "256"
