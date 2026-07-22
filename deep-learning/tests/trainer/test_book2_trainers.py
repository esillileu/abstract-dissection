from __future__ import annotations

import numpy as np

from mlprosection import Tensor
from mlprosection.nn.model import Word2Vec
from mlprosection.nn.model.recurrent import Seq2seq, VanillaRnnlm
from mlprosection.optim.SGD import SGD
from mlprosection.trainer import LanguageModelTrainer, Seq2seqTrainer, Word2VecTrainer


class Receiver:
    def __init__(self) -> None:
        self.updates = []
        self.sources = []
        self.epochs = []
        self.ends = []

    def on_update(self, event) -> None:
        self.updates.append(event)

    def on_source_objective(self, event) -> None:
        self.sources.append(event)

    def on_epoch(self, event) -> None:
        self.epochs.append(event)

    def on_train_end(self, event) -> None:
        self.ends.append(event)


def test_word2vec_trainer_emits_update_and_source_objective_events() -> None:
    model = Word2Vec(8, 3, objective="full_softmax")
    receiver = Receiver()
    trainer = Word2VecTrainer(
        model, SGD(list(model.named_parameters()), lr=0.1),
        max_epochs=1, batch_size=2, event_receivers=[receiver],
    )
    contexts = Tensor(np.asarray([[1, 2], [2, 3], [3, 4], [4, 5]]))
    targets = Tensor(np.asarray([3, 4, 5, 6]))

    trainer.fit(contexts, targets)

    assert [event.update for event in receiver.updates] == [1, 2]
    assert [event.local_iteration for event in receiver.sources] == [0, 1]
    assert receiver.ends[-1].reason == "completed"


def test_word2vec_negative_sampling_reuses_update_candidates_for_post_loss() -> None:
    model = Word2Vec(8, 3, objective="negative_sampling", negative_samples=2)
    receiver = Receiver()
    trainer = Word2VecTrainer(
        model, SGD(list(model.named_parameters()), lr=0.1),
        max_epochs=1, batch_size=2, event_receivers=[receiver],
    )

    trainer.fit(Tensor(np.asarray([[1, 2], [2, 3]])), Tensor(np.asarray([3, 4])))

    assert len(receiver.updates) == 1
    assert len(receiver.sources) == 1


def test_language_model_trainer_emits_bptt_events_and_evaluates_ppl() -> None:
    model = VanillaRnnlm(8, 3, 4)
    receiver = Receiver()
    trainer = LanguageModelTrainer(
        model, SGD(list(model.named_parameters()), lr=0.1),
        max_epochs=1, batch_size=2, time_size=2, event_receivers=[receiver],
    )
    xs = Tensor(np.asarray([1, 2, 3, 4, 5, 6, 1, 2]))
    ts = Tensor(np.asarray([2, 3, 4, 5, 6, 1, 2, 3]))

    trainer.fit(xs, ts)
    result = trainer.evaluate(xs, ts)

    assert len(receiver.updates) == 2
    assert [event.unit_count for event in receiver.sources] == [4, 4]
    assert result.unit == "token"
    assert result.unit_count == len(xs)
    assert result.perplexity is not None


def test_seq2seq_trainer_evaluates_greedy_exact_match() -> None:
    model = Seq2seq(8, 3, 4)
    trainer = Seq2seqTrainer(
        model, SGD(list(model.named_parameters()), lr=0.1),
        max_epochs=1, batch_size=2, start_id=0,
    )
    xs = Tensor(np.asarray([[1, 2, 3], [2, 3, 4]]))
    ts = Tensor(np.asarray([[0, 4, 5], [0, 5, 6]]))

    trainer.fit(xs, ts)
    result = trainer.evaluate(xs, ts, metrics=("exact_match_accuracy", "token_accuracy"))

    assert result.unit == "sequence"
    assert result.unit_count == 2
    assert result.exact_match_accuracy is not None
    assert result.token_accuracy is not None
