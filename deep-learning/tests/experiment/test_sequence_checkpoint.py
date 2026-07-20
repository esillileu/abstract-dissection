from __future__ import annotations

import numpy as np

from mlprosection import Tensor
from mlprosection.experiment.checkpoint import load_epoch_checkpoint, save_epoch_checkpoint
from mlprosection.nn.model.recurrent import Rnnlm
from mlprosection.optim.SGD import Adam
from mlprosection.trainer import TimeTrainer


def test_epoch_checkpoint_restores_stateful_recurrent_state(tmp_path) -> None:
    model = Rnnlm(vocab_size=5, wordvec_size=3, hidden_size=4, backend="cpu")
    optimizer = Adam(list(model.named_parameters()))
    trainer = TimeTrainer(
        model, optimizer, max_epoch=1, batch_size=1, time_size=3, log_interval=1
    )
    model.forward(
        Tensor(np.array([[0]], dtype=np.int64), backend="cpu"),
        Tensor(np.array([[1]], dtype=np.int64), backend="cpu"),
    )
    trainer.epoch = 1
    trainer.time_index = 3

    path = save_epoch_checkpoint(
        root=tmp_path,
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        config_digest="sequence-test",
    )

    restored_model = Rnnlm(vocab_size=5, wordvec_size=3, hidden_size=4, backend="cpu")
    restored_optimizer = Adam(list(restored_model.named_parameters()))
    restored_trainer = TimeTrainer(
        restored_model,
        restored_optimizer,
        max_epoch=1,
        batch_size=1,
        time_size=3,
        log_interval=1,
    )
    load_epoch_checkpoint(
        path=path,
        model=restored_model,
        optimizer=restored_optimizer,
        trainer=restored_trainer,
        config_digest="sequence-test",
    )

    assert restored_trainer.time_index == 3
    assert np.array_equal(restored_model.lstm_layer.h, model.lstm_layer.h)
    assert np.array_equal(restored_model.lstm_layer.c, model.lstm_layer.c)
