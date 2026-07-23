from __future__ import annotations

from pathlib import Path

import numpy as np

from exp.ds1.executor import (
    SupervisedClassificationExecutor,
    get_observation_executor,
)
from exp.ds1.spec import parse_run_spec
from mlprosection import Tensor
from mlprosection.experiment import ExperimentContext


def _context(root: Path) -> ExperimentContext:
    return ExperimentContext(
        metadata={
            "artifact_root": root,
            "checkpoint_root": root / "checkpoints",
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
        "exp.ds1.executor.load_mnist",
        lambda *, flatten, gpu: ((train_x, train_t), (test_x, test_t)),
    )
    spec = parse_run_spec(
        "exp/ds1/config/e01_mnist_optimizer.yaml",
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

    result = SupervisedClassificationExecutor().run(
        config,
        _context(tmp_path),
    )

    assert result.metrics["final/status/success"] == 1.0
    assert result.metrics["final/system/total_updates"] == 1.0
    assert (tmp_path / "updates.csv").is_file()


def test_ds1_optimizer_observation_config_runs(
    tmp_path: Path,
) -> None:
    spec = parse_run_spec(
        "exp/ds1/config/e09_optimizer_trajectory.yaml",
        atomic_run_id="TOY-SGD",
        overrides={
            "budget": {"max_updates": 2},
            "tracking": {"enabled": False},
        },
    )
    config = spec.to_executor_config()
    config["seed"] = 1

    result = get_observation_executor(config).run(
        config,
        _context(tmp_path),
    )

    assert result.metrics["final/system/total_updates"] == 2.0
    assert (tmp_path / "observations" / "trajectory.csv").is_file()


def test_ds1_activation_observation_config_runs(
    tmp_path: Path,
) -> None:
    spec = parse_run_spec(
        "exp/ds1/config/e10_activation_observation.yaml",
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
