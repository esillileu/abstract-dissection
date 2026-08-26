from __future__ import annotations

from deepscratch.core import Tensor
from deepscratch.nn.model import Model
from deepscratch.nn.objective import SoftmaxCrossEntropy
from deepscratch.trainer import ForwardTrainer
from deepscratch.trainer.events import EpochEvent, TrainEndEvent, UpdateEvent


class IdentityModel(Model):
    def forward_manual(self, x, *, cache=True):
        return x

    def backward_manual(self, dout):
        return dout


class DummyOptimizer:
    def __init__(self) -> None:
        self.params = []
        self.lr = 0.1
        self.update_count = 0

    def update(self) -> None:
        self.update_count += 1


class Recorder:
    def __init__(self) -> None:
        self.updates: list[UpdateEvent] = []
        self.epochs: list[EpochEvent] = []
        self.ends: list[TrainEndEvent] = []

    def on_update(self, event: UpdateEvent) -> None:
        self.updates.append(event)

    def on_epoch(self, event: EpochEvent) -> None:
        self.epochs.append(event)

    def on_train_end(self, event: TrainEndEvent) -> None:
        self.ends.append(event)


def test_forward_trainer_emits_one_post_update_event_per_successful_update() -> None:
    recorder = Recorder()
    optimizer = DummyOptimizer()
    trainer = ForwardTrainer(
        IdentityModel(),
        SoftmaxCrossEntropy(),
        optimizer,
        max_epochs=2,
        max_updates=3,
        batch_size=2,
        event_receivers=[recorder],
    )
    x = Tensor([[0.1, 0.9], [0.8, 0.2], [0.7, 0.3], [0.2, 0.8]])
    t = Tensor([1, 0, 0, 1])

    trainer.fit(x, t)

    assert optimizer.update_count == 3
    assert [event.update for event in recorder.updates] == [1, 2, 3]
    assert [event.epoch for event in recorder.updates] == [1, 1, 2]
    assert [event.batch_size for event in recorder.updates] == [2, 2, 2]
    assert all(event.loss.data > 0 for event in recorder.updates)
    assert [(event.start_update, event.end_update) for event in recorder.epochs] == [
        (1, 2),
        (3, 3),
    ]
    assert recorder.ends == [TrainEndEvent(reason="max_updates", update=3, epoch=2)]


def test_forward_trainer_records_remainder_batch_and_restores_state() -> None:
    recorder = Recorder()
    trainer = ForwardTrainer(
        IdentityModel(),
        SoftmaxCrossEntropy(),
        DummyOptimizer(),
        max_epochs=1,
        batch_size=2,
        event_receivers=[recorder],
    )
    x = Tensor([[0.1, 0.9], [0.8, 0.2], [0.7, 0.3]])
    t = Tensor([1, 0, 0])

    trainer.fit(x, t)
    state = trainer.state_dict()
    restored = ForwardTrainer(
        IdentityModel(),
        SoftmaxCrossEntropy(),
        DummyOptimizer(),
        max_epochs=2,
        batch_size=2,
    )
    restored.load_state_dict(state)

    assert [event.batch_size for event in recorder.updates] == [2, 1]
    assert state == {"global_step": 2, "epoch": 1}
    assert restored.global_step == 2
    assert restored.epoch == 1
