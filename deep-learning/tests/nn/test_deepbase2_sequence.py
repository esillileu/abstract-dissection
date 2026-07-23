from __future__ import annotations

import numpy as np

from mlprosection import Tensor
from mlprosection.nn.layers import TimeAttention
from mlprosection.nn.model import TimeSequential
from mlprosection.nn.model.architecture import AttentionSeq2seq, CBOW, PeekySeq2seq, VanillaRnnlm
from mlprosection.nn.objective import FullSoftmax, NegativeSampling, TemporalSoftmaxCrossEntropy
from mlprosection.optim.SGD import Adam
from mlprosection.optim.transform import ClipGradNorm
from mlprosection.trainer import LanguageModelTrainer


def test_word2vec_objectives_populate_embedding_gradients() -> None:
    contexts = Tensor(np.array([[0, 1], [1, 2], [2, 3]]), backend="cpu")
    targets = Tensor(np.array([2, 3, 4]), backend="cpu")
    for objective in (
        FullSoftmax(6, 4, backend="cpu"),
        NegativeSampling(6, 4, negative_samples=2, backend="cpu"),
    ):
        model = CBOW(6, 4, backend="cpu")
        result = objective.forward(model.forward(contexts), targets)
        model.backward(objective.backward())
        assert float(result.loss.data) > 0
        assert np.any(model.W_in.grad)
        assert np.any(objective.W_out.grad)


def test_language_model_trainer_runs_truncated_bptt() -> None:
    model = VanillaRnnlm(vocab_size=7, wordvec_size=4, hidden_size=5, backend="cpu")
    objective = TemporalSoftmaxCrossEntropy()
    params = [
        *((f"model.{name}", value) for name, value in model.named_parameters()),
        *((f"objective.{name}", value) for name, value in objective.named_parameters()),
    ]
    trainer = LanguageModelTrainer(
        model,
        objective,
        Adam(
            params,
            pre_step_hooks=[ClipGradNorm(0.25)],
        ),
        max_epochs=1,
        batch_size=2,
        time_size=3,
    )
    tokens = Tensor(np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4]), backend="cpu")

    trainer.fit(tokens[:-1], tokens[1:])

    assert trainer.global_step > 0
    assert trainer.evaluate(tokens[:-1], tokens[1:]).perplexity > 0


def test_time_sequential_resets_nested_state() -> None:
    model = VanillaRnnlm(vocab_size=7, wordvec_size=4, hidden_size=5, backend="cpu")
    container = TimeSequential(model.rnn_layer)
    container.reset_runtime_state()
    assert model.rnn_layer.h is None


def test_peeky_and_attention_seq2seq_support_training_and_decode() -> None:
    xs = Tensor(np.array([[1, 2, 3], [3, 2, 1]]), backend="cpu")
    ts = Tensor(np.array([[0, 1, 2, 3], [0, 3, 2, 1]]), backend="cpu")
    for model_class in (PeekySeq2seq, AttentionSeq2seq):
        model = model_class(vocab_size=6, wordvec_size=3, hidden_size=4, backend="cpu")
        objective = TemporalSoftmaxCrossEntropy()
        result = objective.forward(model.forward(xs, ts[:, :-1]), ts[:, 1:])
        model.backward(objective.backward())
        decoded = model.generate(xs[:1], start_id=0, sample_size=3)
        assert result.loss.shape == ()
        assert len(decoded) == 3
    attention_model = AttentionSeq2seq(
        vocab_size=6, wordvec_size=3, hidden_size=4, backend="cpu"
    )
    assert isinstance(attention_model.decoder.attention, TimeAttention)
