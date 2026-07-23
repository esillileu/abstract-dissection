from __future__ import annotations

from pathlib import Path

from mlprosection.nn.model.architecture import MLP
from mlprosection.nn.objective import SoftmaxCrossEntropy
from mlprosection.optim.SGD import SGD
from mlprosection.experiment.checkpoint import (
    CheckpointManager,
    CheckpointRetentionPolicy,
    load_epoch_checkpoint,
)


class _Trainer:
    def __init__(self) -> None:
        self.epoch = 0
        self.global_step = 0

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "global_step": self.global_step}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.epoch = int(state["epoch"])
        self.global_step = int(state["global_step"])


def _manager(root: Path, *, policy: CheckpointRetentionPolicy | None = None):
    model = MLP(input_size=2, hidden_sizes=[3], output_size=2)
    objective = SoftmaxCrossEntropy()
    params = [
        *((f"model.{name}", value) for name, value in model.named_parameters()),
        *((f"objective.{name}", value) for name, value in objective.named_parameters()),
    ]
    optimizer = SGD(params, lr=0.1)
    trainer = _Trainer()
    manager = CheckpointManager(
        root=root,
        model=model,
        objective=objective,
        optimizer=optimizer,
        trainer=trainer,
        config_digest="config",
        policy=policy,
    )
    return manager, model, objective, optimizer, trainer


def test_best_and_latest_each_keep_one_generation(tmp_path) -> None:
    manager, _model, _objective, _optimizer, trainer = _manager(tmp_path)
    for epoch in range(1, 4):
        trainer.epoch = epoch
        trainer.global_step = epoch * 10
        manager.save_latest()
        manager.save_best()

    generations = tmp_path / "generations"
    assert len(list(generations.glob("latest-*"))) == 1
    assert len(list(generations.glob("best-*"))) == 1
    assert manager.current("latest").epoch == 3
    assert manager.current("best").epoch == 3
    assert (tmp_path / "latest.json").is_file()
    assert (tmp_path / "best.json").is_file()


def test_periodic_generations_require_explicit_policy_and_keep_limit(tmp_path) -> None:
    policy = CheckpointRetentionPolicy(periodic_every_epochs=2, periodic_keep=2)
    manager, _model, _objective, _optimizer, trainer = _manager(tmp_path, policy=policy)
    for epoch in range(1, 7):
        trainer.epoch = epoch
        trainer.global_step = epoch * 10
        manager.save_periodic_if_due()

    retained = manager.retained_periodic()
    assert [ref.epoch for ref in retained] == [4, 6]


def test_checkpoint_pointer_can_be_loaded(tmp_path) -> None:
    manager, model, objective, optimizer, trainer = _manager(tmp_path)
    trainer.epoch = 2
    trainer.global_step = 20
    manager.save_latest()
    generation = manager.current("latest").path
    assert (generation / "model_parameters.npz").is_file()
    assert (generation / "model_buffers.npz").is_file()
    assert (generation / "objective_parameters.npz").is_file()
    assert (generation / "objective_buffers.npz").is_file()
    assert (generation / "optimizer_state.pkl").is_file()
    assert (generation / "trainer_state.pkl").is_file()
    assert (generation / "rng_state.pkl").is_file()
    trainer.epoch = 9
    trainer.global_step = 90

    load_epoch_checkpoint(
        path=tmp_path / "latest.json",
        model=model,
        objective=objective,
        optimizer=optimizer,
        trainer=trainer,
        config_digest="config",
    )

    assert trainer.epoch == 2
    assert trainer.global_step == 20
