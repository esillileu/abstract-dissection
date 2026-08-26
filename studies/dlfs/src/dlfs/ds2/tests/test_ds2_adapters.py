"""Unit tests for DS2 DeepScratch representation adapters."""

from __future__ import annotations

import numpy as np
from deepscratch.nn.model.architecture import (
    CBOW,
    AttentionSeq2seq,
    CBOWBatchAdapter,
    OneHotCBOWBatchAdapter,
    OneHotSkipGramBatchAdapter,
    PairExpandedSkipGramBatchAdapter,
    PeekySeq2seq,
    Rnnlm,
    Seq2seq,
    SkipGram,
    SkipGramBatchAdapter,
    TiedRnnlm,
    VanillaRnnlm,
)
from deepscratch.nn.objective import (
    FusedNegativeSampling,
    NegativeSampling,
    SoftmaxWithLoss,
    TemporalSoftmaxCrossEntropy,
)
from deepscratch.optim.SGD import SGD, Adam
from deepscratch.optim.transform import ClipGradNorm

from dlfs.ds2.implemented.adapters import (
    build_language_model,
    build_seq2seq_model,
    build_sequence_optimizer,
    build_word2vec_batch_adapter,
    build_word2vec_model,
    build_word2vec_objective,
    contexts_targets,
    language_model_training_corpus,
)


def test_build_word2vec_model() -> None:
    cbow = build_word2vec_model("CBOW", "embedding", 100, 16, "cpu")
    assert isinstance(cbow, CBOW)

    skipgram = build_word2vec_model("SkipGram", "embedding", 100, 16, "cpu")
    assert isinstance(skipgram, SkipGram)


def test_build_word2vec_batch_adapter() -> None:
    assert isinstance(
        build_word2vec_batch_adapter("CBOW", "embedding", 100, "NegativeSampling"),
        CBOWBatchAdapter,
    )
    assert isinstance(
        build_word2vec_batch_adapter("SkipGram", "embedding", 100, "NegativeSampling"),
        SkipGramBatchAdapter,
    )
    assert isinstance(
        build_word2vec_batch_adapter("CBOW", "one_hot", 100, "SoftmaxWithLoss"),
        OneHotCBOWBatchAdapter,
    )
    assert isinstance(
        build_word2vec_batch_adapter("SkipGram", "one_hot", 100, "SoftmaxWithLoss"),
        OneHotSkipGramBatchAdapter,
    )
    assert isinstance(
        build_word2vec_batch_adapter(
            "DumbSkipGram", "embedding", 100, "SoftmaxWithLoss"
        ),
        PairExpandedSkipGramBatchAdapter,
    )


def test_build_word2vec_objective() -> None:
    softmax = build_word2vec_objective(
        "SoftmaxWithLoss", {"reduction": "mean"}, 100, None, "cpu"
    )
    assert isinstance(softmax, SoftmaxWithLoss)

    ns = build_word2vec_objective(
        "NegativeSampling", {"negative_samples": 5}, 100, None, "cpu"
    )
    assert isinstance(ns, NegativeSampling)

    fns = build_word2vec_objective(
        "FusedNegativeSampling", {"negative_samples": 5}, 100, None, "cpu"
    )
    assert isinstance(fns, FusedNegativeSampling)


def test_build_language_model_and_seq2seq_model() -> None:
    vanilla = build_language_model(
        "VanillaRnnlm", 50, {"wordvec_size": 10, "hidden_size": 20}, "cpu"
    )
    assert isinstance(vanilla, VanillaRnnlm)

    rnnlm = build_language_model(
        "Rnnlm", 50, {"wordvec_size": 10, "hidden_size": 20}, "cpu"
    )
    assert isinstance(rnnlm, Rnnlm)

    tied = build_language_model(
        "TiedRnnlm", 50, {"wordvec_size": 10, "hidden_size": 10}, "cpu"
    )
    assert isinstance(tied, TiedRnnlm)

    seq = build_seq2seq_model(
        "Seq2seq", 20, {"wordvec_size": 8, "hidden_size": 16}, "cpu"
    )
    assert isinstance(seq, Seq2seq)

    peeky = build_seq2seq_model(
        "PeekySeq2seq", 20, {"wordvec_size": 8, "hidden_size": 16}, "cpu"
    )
    assert isinstance(peeky, PeekySeq2seq)

    attn = build_seq2seq_model(
        "AttentionSeq2seq", 20, {"wordvec_size": 8, "hidden_size": 16}, "cpu"
    )
    assert isinstance(attn, AttentionSeq2seq)


def test_build_sequence_optimizer() -> None:
    model = Seq2seq(vocab_size=10, wordvec_size=4, hidden_size=8, backend="cpu")
    obj = TemporalSoftmaxCrossEntropy(backend="cpu")

    adam = build_sequence_optimizer(
        {
            "optimizer": {"name": "adam", "learning_rate": 0.005},
            "policy": {"max_grad": 5.0},
        },
        model,
        obj,
    )
    assert isinstance(adam, Adam)
    assert adam.lr == 0.005
    assert len(adam.pre_step_hooks) == 1
    assert isinstance(adam.pre_step_hooks[0], ClipGradNorm)
    assert adam.pre_step_hooks[0].max_norm == 5.0

    sgd = build_sequence_optimizer(
        {"optimizer": {"name": "sgd", "learning_rate": 1.5}},
        model,
        obj,
    )
    assert isinstance(sgd, SGD)
    assert sgd.lr == 1.5


def test_contexts_targets_and_language_model_training_corpus() -> None:
    corpus = np.arange(10, dtype=np.int64)
    contexts, targets = contexts_targets(corpus, 1)
    assert contexts.shape == (8, 2)
    assert targets.shape == (8,)

    sliced, vocab_size = language_model_training_corpus(corpus, {"train_limit": 6})
    assert len(sliced) == 6
    assert vocab_size == 6
