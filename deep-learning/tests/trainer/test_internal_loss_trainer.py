from __future__ import annotations

import numpy as np

from mlprosection import Tensor
from mlprosection.nn.model import Word2Vec
from mlprosection.optim.SGD import Adam
from mlprosection.profiling import ProfilingConfig
from mlprosection.trainer import InternalLossTrainer


def test_internal_loss_trainer_owns_shuffled_update_loop_and_state() -> None:
    contexts = Tensor(np.array([[0, 1], [1, 2], [2, 3], [3, 4]]), backend="cpu")
    targets = Tensor(np.array([2, 3, 4, 5]), backend="cpu")
    model = Word2Vec(6, 3, objective="full_softmax", backend="cpu")
    trainer = InternalLossTrainer(
        model,
        Adam(list(model.named_parameters())),
        max_epoch=2,
        batch_size=2,
        log_interval=1,
        max_updates=3,
        profiling_config=ProfilingConfig(enabled=True),
    )

    history = trainer.fit(contexts, targets)

    assert trainer.global_step == 3
    assert trainer.epoch == 2
    assert len(history.interval_loss) == 3
    assert len(history.epoch_loss) == 2
    assert trainer.state_dict()["global_step"] == 3
    metrics = trainer.profiling_metrics()
    assert metrics["model.parameter_count"] > 0
    assert metrics["runtime.epoch.0.train_duration_ms"] >= 0
    assert metrics["throughput.epoch.0.train_samples_per_s"] >= 0
