from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from deepscratch.core import Tensor

from dlfs.ds1.implemented.executor import (
    SupervisedClassificationExecutor,
    get_observation_executor,
)
from dlfs.ds1.implemented.spec import parse_run_spec
from repro_core.context import ExperimentContext


class _ProgressRecorder:
    def __init__(self) -> None:
        self.total: int | None = None
        self.updates: list[int] = []

    def set_total_updates(self, total: int, *, completed: int = 0) -> None:
        self.total = total

    def advance_to(self, update: int, *, epoch: int = 0) -> None:
        self.updates.append(update)

    def on_update(self, event) -> None:
        self.advance_to(event.update, epoch=event.epoch)

    def on_epoch(self, _event) -> None:
        return None

    def on_train_end(self, _event) -> None:
        return None


def _context(root: Path, progress=None) -> ExperimentContext:
    return ExperimentContext(
        metadata={
            "artifact_root": root,
            "checkpoint_root": root / "checkpoints",
            **({} if progress is None else {"progress_reporter": progress}),
        }
    )


def test_ds1_supervised_config_runs_one_update(
    monkeypatch,
    tmp_path: Path,
) -> None:
    train_x = Tensor(np.zeros((4, 784), dtype=np.float64), backend="cpu")
    train_t = Tensor(np.array([0, 1, 0, 1]), backend="cpu")
    test_x = Tensor(np.zeros((2, 784), dtype=np.float64), backend="cpu")
    test_t = Tensor(np.array([0, 1]), backend="cpu")
    monkeypatch.setattr(
        "dlfs.ds1.implemented.executor.load_mnist",
        lambda *, flatten, gpu, **kwargs: ((train_x, train_t), (test_x, test_t)),
    )
    spec = parse_run_spec(
        "studies/dlfs/src/dlfs/ds1/config/implemented/e01_mnist_optimizer.yaml",
        atomic_run_id="MLP-OPT-SGD",
        overrides={
            "budget": {"max_epochs": 1, "max_updates": 1},
            "loader": {"batch_size": 2},
            "recording": {"evaluation_sources": [], "triggers": []},
            "numerics": {
                "backend": "numpy",
                "device": "cpu",
                "dtype": "float64",
            },
            "tracking": {"enabled": False},
        },
    )
    config = spec.to_executor_config()
    config["seed"] = 1
    progress = _ProgressRecorder()

    result = SupervisedClassificationExecutor().run(
        config,
        _context(tmp_path, progress),
    )

    assert result.metrics["final/status/success"] == 1.0
    assert result.metrics["final/system/total_updates"] == 1.0
    assert progress.total == 1
    assert progress.updates == [1]
    assert (tmp_path / "updates.csv").is_file()


def test_extended_mlp_with_batchnorm_dropout_and_weight_decay_runs_one_update(
    monkeypatch,
    tmp_path: Path,
) -> None:
    train_x = Tensor(np.zeros((4, 784), dtype=np.float64), backend="cpu")
    train_t = Tensor(np.array([0, 1, 0, 1]), backend="cpu")
    test_x = Tensor(np.zeros((2, 784), dtype=np.float64), backend="cpu")
    test_t = Tensor(np.array([0, 1]), backend="cpu")
    monkeypatch.setattr(
        "dlfs.ds1.implemented.executor.load_mnist",
        lambda *, flatten, gpu, **kwargs: ((train_x, train_t), (test_x, test_t)),
    )
    spec = parse_run_spec(
        "studies/dlfs/src/dlfs/ds1/config/implemented/e12_mnist_extended_mlp.yaml",
        atomic_run_id="MLP-EXT-ALL-BOOK",
        overrides={
            "budget": {"max_epochs": 1, "max_updates": 1},
            "loader": {"batch_size": 2},
            "recording": {"evaluation_sources": [], "triggers": []},
            "numerics": {
                "backend": "numpy",
                "device": "cpu",
                "dtype": "float64",
            },
            "tracking": {"enabled": False},
        },
    )
    config = spec.to_executor_config()
    config["seed"] = 1

    result = SupervisedClassificationExecutor().run(
        config,
        _context(tmp_path),
    )

    assert result.metrics["final/status/success"] == 1.0
    assert result.metrics["final/system/total_updates"] == 1.0
    assert "final/train-full/accuracy" in result.metrics
    assert "final/train-test/accuracy-gap" in result.metrics
    assert result.metrics["final/train-test/accuracy-gap"] == pytest.approx(
        result.metrics["final/train-full/accuracy"]
        - result.metrics["final/test/accuracy"]
    )
    assert (tmp_path / "updates.csv").is_file()


def test_ds1_optimizer_observation_config_runs(
    tmp_path: Path,
) -> None:
    spec = parse_run_spec(
        "studies/dlfs/src/dlfs/ds1/config/implemented/e09_optimizer_trajectory.yaml",
        atomic_run_id="TOY-SGD",
        overrides={
            "budget": {"max_updates": 2},
            "tracking": {"enabled": False},
        },
    )
    config = spec.to_executor_config()
    config["seed"] = 1
    progress = _ProgressRecorder()

    result = get_observation_executor(config).run(
        config,
        _context(tmp_path, progress),
    )

    assert result.metrics["final/system/total_updates"] == 2.0
    assert progress.total == 2
    assert progress.updates == [1, 2]
    assert (tmp_path / "observations" / "trajectory.csv").is_file()


def test_ds1_activation_observation_config_runs(
    tmp_path: Path,
) -> None:
    spec = parse_run_spec(
        "studies/dlfs/src/dlfs/ds1/config/implemented/e10_activation_observation.yaml",
        atomic_run_id="ACT-RELU-HE",
        overrides={
            "model": {"width": 4, "depth": 2},
            "tracking": {"enabled": False},
        },
    )
    config = spec.to_executor_config()
    config["seed"] = 1

    result = get_observation_executor(config).run(
        config,
        _context(tmp_path),
    )

    assert result.metrics["final/status/success"] == 1.0
    assert (tmp_path / "observations" / "activation_histogram.csv").is_file()
    assert (tmp_path / "observations" / "activation_summary.csv").is_file()


def test_ds1_gradient_check_observation_runs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    train_x = Tensor(np.zeros((3, 4), dtype=np.float64), backend="cpu")
    train_t = Tensor(np.eye(2, dtype=np.float64)[[0, 1, 0]], backend="cpu")
    test_x = Tensor(np.zeros((1, 4), dtype=np.float64), backend="cpu")
    test_t = Tensor(np.eye(2, dtype=np.float64)[[0]], backend="cpu")
    monkeypatch.setattr(
        "dlfs.ds1.implemented.executor.load_mnist",
        lambda **_kwargs: ((train_x, train_t), (test_x, test_t)),
    )
    spec = parse_run_spec(
        "studies/dlfs/src/dlfs/ds1/config/implemented/e14_gradient_check.yaml",
        atomic_run_id="TWO-LAYER-GRADIENT-CHECK",
        overrides={
            "model": {"input_size": 4, "hidden_size": 3, "output_size": 2},
            "tracking": {"enabled": False},
        },
    )
    config = spec.to_executor_config()
    config["seed"] = 1
    progress = _ProgressRecorder()

    result = get_observation_executor(config).run(
        config,
        _context(tmp_path, progress),
    )

    assert result.metrics["final/status/success"] == 1.0
    assert progress.updates == [1, 2]
    assert (tmp_path / "observations" / "gradient_check.csv").is_file()
    assert (tmp_path / "observations" / "gradient_timing.csv").is_file()
    assert (tmp_path / "observations" / "gradients.npz").is_file()
