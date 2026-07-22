from __future__ import annotations

from mlprosection import Tensor
from mlprosection.experiment import ExperimentContext
from exp.ds1.executor import SupervisedClassificationExecutor


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
        "dataset": {"flatten": True, "train_evaluation_size": 2},
        "model": {
            "alias": "MLP", "input_size": 2, "hidden_sizes": [3],
            "output_size": 2, "activation": "relu", "initializer": "he",
        },
        "optimizer": {"name": "sgd", "learning_rate": 0.01},
        "training": {
            "max_epochs": 1, "max_updates": 2,
            "record_first_validation_evaluation": True,
            "record_step_validation_interval": 1,
            "record_step_train_evaluation": True,
        },
        "loader": {"batch_size": 2, "drop_last": False, "sampling_method": "permutation_per_epoch"},
        "checkpoint": {"save_on_eval": False},
        "profiling": {},
        "numerics": {"backend": "numpy", "device": "cpu", "dtype": "float64"},
    }
    context = ExperimentContext(metadata={"artifact_root": tmp_path, "checkpoint_root": tmp_path / "checkpoints"})

    result = SupervisedClassificationExecutor().run(config, context)

    update_rows = [row for row in result.history if row[0] == "update" and row[2] == "train/loss"]
    timing_rows = [row for row in result.history if row[2] == "runtime/window/train_wall_time_ms"]
    evaluation_rows = [row for row in result.history if row[0] == "update" and row[2].startswith("eval_train/")]
    assert [row[1] for row in update_rows] == [1, 2]
    assert [row[1] for row in timing_rows] == [1, 2]
    assert {row[2] for row in evaluation_rows} == {
        "eval_train/loss", "eval_train/accuracy",
    }
    assert result.metrics["final/system/total_updates"] == 2.0
