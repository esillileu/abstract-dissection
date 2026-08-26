"""Unit tests for DeepScratch checkpoint adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from deepscratch.nn.model.architecture import TwoLayerNet
from deepscratch.nn.objective import SoftmaxCrossEntropy
from deepscratch.optim.SGD import SGD
from deepscratch.trainer import ForwardTrainer

from dlfs.adapters.checkpoint import (
    create_deepscratch_checkpoint_manager,
    load_deepscratch_checkpoint,
    load_deepscratch_model_parameters,
    write_deepscratch_checkpoint,
)
from repro_core.context.checkpoint import (
    CheckpointRetentionPolicy,
    resolve_checkpoint_path,
)


def test_checkpoint_adapter_roundtrips_deepscratch_state(tmp_path: Path) -> None:
    model = TwoLayerNet(input_size=4, hidden_size=3, output_size=2)
    objective = SoftmaxCrossEntropy()
    optimizer = SGD(model.named_parameters(), lr=0.01)
    trainer = ForwardTrainer(model, objective, optimizer, max_epochs=5, batch_size=2)
    trainer.epoch = 2
    trainer.global_step = 40

    ckpt_dir = tmp_path / "ckpt_gen"
    write_deepscratch_checkpoint(
        ckpt_dir,
        model=model,
        objective=objective,
        optimizer=optimizer,
        trainer=trainer,
        config_digest="abc123digest",
    )

    assert (ckpt_dir / "model_parameters.npz").is_file()
    assert (ckpt_dir / "manifest.json").is_file()

    # Save original weights
    orig_w1 = model.layers[0].W.data.copy()

    # Mutate weights
    model.layers[0].W.data += 10.0
    trainer.epoch = 0
    trainer.global_step = 0

    # Restore via load_deepscratch_checkpoint
    load_deepscratch_checkpoint(
        path=ckpt_dir,
        model=model,
        objective=objective,
        optimizer=optimizer,
        trainer=trainer,
        config_digest="abc123digest",
    )

    np.testing.assert_allclose(model.layers[0].W.data, orig_w1)
    assert trainer.epoch == 2
    assert trainer.global_step == 40


def test_checkpoint_manager_factory_integrates_with_repro_core(
    tmp_path: Path,
) -> None:
    model = TwoLayerNet(input_size=4, hidden_size=3, output_size=2)
    objective = SoftmaxCrossEntropy()
    optimizer = SGD(model.named_parameters(), lr=0.01)
    trainer = ForwardTrainer(model, objective, optimizer, max_epochs=5, batch_size=2)
    trainer.epoch = 1
    trainer.global_step = 10

    manager = create_deepscratch_checkpoint_manager(
        tmp_path / "checkpoints",
        model=model,
        objective=objective,
        optimizer=optimizer,
        trainer=trainer,
        config_digest="dig1",
        policy=CheckpointRetentionPolicy(periodic_every_epochs=1, periodic_keep=2),
    )

    latest_ref = manager.save_latest()
    assert latest_ref.role == "latest"
    assert (tmp_path / "checkpoints" / "latest.json").is_file()
    resolved = resolve_checkpoint_path(tmp_path / "checkpoints" / "latest.json")
    assert resolved.is_dir()
    assert (resolved / "model_parameters.npz").is_file()

    periodic_ref = manager.save_periodic_if_due()
    assert periodic_ref is not None
    assert periodic_ref.role == "periodic"

    # Test load_deepscratch_model_parameters on pointer
    model_restored = TwoLayerNet(input_size=4, hidden_size=3, output_size=2)
    load_deepscratch_model_parameters(
        tmp_path / "checkpoints" / "latest.json", model_restored
    )
    np.testing.assert_allclose(model_restored.layers[0].W.data, model.layers[0].W.data)
