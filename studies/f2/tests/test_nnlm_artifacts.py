"""NNLM resume and shared evaluation boundary, using test-owned corpora only."""

import json

import numpy as np
import pytest
from w2v import (
    Corpus,
    DownpourConfig,
    NnlmDownpourTrainingSession,
    NnlmTrainingConfig,
    NnlmTrainingSession,
    Vocabulary,
    VocabularyConfig,
)

from f2.suites.w2v.artifacts import load_lookup_artifact, save_lookup_artifact
from f2.suites.w2v.nnlm_artifacts import (
    create_checkpoint_manager,
    load_checkpoint,
    save_checkpoint,
)


def fixture(tmp_path, **overrides):
    path = tmp_path / "corpus.txt"
    path.write_text("a b a c d a b\nb a c b d a")
    corpus = Corpus(path)
    vocabulary = Vocabulary.build(
        corpus, VocabularyConfig(initial_capacity=8, hash_capacity=31, min_count=1)
    )
    values = dict(
        embedding_dimension=2,
        history_length=2,
        hidden_dimension=3,
        hidden_activation="tanh",
        epochs=2,
        thread_count=1,
        initial_learning_rate=0.1,
        root_seed=19,
    )
    values.update(overrides)
    return corpus, vocabulary, NnlmTrainingConfig(**values)


def test_checkpoint_resume_matches_uninterrupted_and_lookup(tmp_path):
    corpus, vocabulary, config = fixture(tmp_path)
    continuous = NnlmTrainingSession(corpus, vocabulary, config)
    continuous.train_epoch()
    continuous.train_epoch()
    interrupted = NnlmTrainingSession(corpus, vocabulary, config)
    interrupted.train_epoch()
    manager = create_checkpoint_manager(tmp_path / "checkpoints", session=interrupted)
    saved = manager.save_latest()
    state = load_checkpoint(
        saved.path, expected={"config_digest": interrupted.export_state().config_digest}
    )
    resumed = NnlmTrainingSession.restore(corpus, vocabulary, config, state)
    assert resumed.completed_epochs == 1
    resumed.train_epoch()
    expected = continuous.export_state()
    actual = resumed.export_state()
    for name in ("input_embeddings", "hidden_weights", "hidden_bias", "output_weights"):
        assert np.array_equal(getattr(expected, name)(), getattr(actual, name)())
    assert actual.processed_tokens == expected.processed_tokens == 26
    lookup_path = tmp_path / "lookup"
    save_lookup_artifact(actual, lookup_path)
    lookup = load_lookup_artifact(lookup_path)
    assert np.array_equal(lookup.embeddings, actual.input_embeddings())
    assert np.array_equal(
        lookup.vector(b"a"), actual.input_embeddings()[lookup.row(b"a")]
    )
    assert lookup.manifest["format"] == "f2-w2v-lookup-v1"


def test_identity_corruption_and_parameter_corruption_are_rejected(tmp_path):
    corpus, vocabulary, config = fixture(tmp_path)
    session = NnlmTrainingSession(corpus, vocabulary, config)
    session.train_epoch()
    state = session.export_state()
    _, _, changed_config = fixture(tmp_path, root_seed=20)
    with pytest.raises(ValueError):
        NnlmTrainingSession.restore(corpus, vocabulary, changed_config, state)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(state, checkpoint)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_checkpoint(checkpoint, expected={"corpus_digest": "wrong"})
    weights = checkpoint / "hidden_weights.npy"
    weights.write_bytes(weights.read_bytes()[:-1] + b"\x00")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_checkpoint(checkpoint)
    (tmp_path / "corpus.txt").write_text("different corpus")
    with pytest.raises(ValueError):
        NnlmTrainingSession.restore(corpus, vocabulary, config, state)


def test_checkpoint_format_does_not_reuse_w2v_schema(tmp_path):
    corpus, vocabulary, config = fixture(tmp_path)
    state = NnlmTrainingSession(corpus, vocabulary, config).export_state()
    save_checkpoint(state, tmp_path / "checkpoint")
    manifest = json.loads((tmp_path / "checkpoint" / "manifest.json").read_text())
    assert manifest["format"] == "f2-nnlm-checkpoint-v1"
    assert manifest["worker_state"] == {
        "boundary": "epoch",
        "stochastic_training": False,
    }
    assert "output_embeddings.npy" not in manifest["files"]


@pytest.mark.parametrize("workers", [1, 3])
def test_downpour_checkpoint_resume_all_parameters_and_optimizer(tmp_path, workers):
    corpus, vocabulary, config = fixture(tmp_path, thread_count=workers, epochs=3)
    downpour = DownpourConfig(
        parameter_server_shards=3, mini_batch_targets=2, queue_capacity=1
    )
    session = NnlmDownpourTrainingSession(corpus, vocabulary, config, downpour)
    first = session.train_epoch()
    assert first.epoch_tokens == 13
    assert first.objective_loss_count == 9
    assert np.isfinite(first.objective_loss)
    state = session.export_state()
    assert state.optimizer_kind == "adagrad"
    assert len(state.replicas()) == workers
    assert sum(replica.processed_tokens for replica in state.replicas()) == 13
    for accumulator in state.adagrad_arrays():
        assert np.isfinite(accumulator).all()
        assert (accumulator >= 0).all()
        assert (accumulator > 0).any()
    manager = create_checkpoint_manager(tmp_path / "checkpoints", session=session)
    saved = manager.save_latest()
    loaded = load_checkpoint(saved.path)
    for original, restored in zip(
        state.adagrad_arrays(), loaded.adagrad_arrays(), strict=True
    ):
        assert np.array_equal(original, restored)
    assert [replica.batch_count for replica in loaded.replicas()] == [
        replica.batch_count for replica in state.replicas()
    ]
    resumed = NnlmDownpourTrainingSession.restore(
        corpus, vocabulary, config, downpour, loaded
    )
    resumed.train_epoch()
    resumed.train_epoch()
    assert resumed.is_complete
    actual = resumed.export_state()
    assert actual.processed_tokens == 39
    if workers == 1:
        session.train_epoch()
        session.train_epoch()
        expected = session.export_state()
        for name in (
            "input_embeddings",
            "hidden_weights",
            "hidden_bias",
            "output_weights",
        ):
            assert np.array_equal(getattr(expected, name)(), getattr(actual, name)())
        for a, b in zip(
            expected.adagrad_arrays(), actual.adagrad_arrays(), strict=True
        ):
            assert np.array_equal(a, b)
        assert (
            expected.replicas()[0].objective_count
            == actual.replicas()[0].objective_count
        )
        assert expected.replicas()[0].batch_count == actual.replicas()[0].batch_count
    lookup_path = tmp_path / "lookup"
    save_lookup_artifact(actual, lookup_path)
    assert np.array_equal(
        load_lookup_artifact(lookup_path).embeddings, actual.input_embeddings()
    )


def test_downpour_checkpoint_requires_matching_optimizer_and_detects_corrupt_accumulator(
    tmp_path,
):
    corpus, vocabulary, config = fixture(tmp_path)
    downpour = DownpourConfig(parameter_server_shards=3, mini_batch_targets=2)
    session = NnlmDownpourTrainingSession(corpus, vocabulary, config, downpour)
    session.train_epoch()
    state = session.export_state()
    with pytest.raises(ValueError, match="Downpour session"):
        NnlmTrainingSession.restore(corpus, vocabulary, config, state)
    sgd = NnlmTrainingSession(corpus, vocabulary, config).export_state()
    with pytest.raises(ValueError, match="AdaGrad and replica state"):
        NnlmDownpourTrainingSession.restore(corpus, vocabulary, config, downpour, sgd)
    changed = DownpourConfig(
        parameter_server_shards=3, mini_batch_targets=2, adagrad_gamma=0.2
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        NnlmDownpourTrainingSession.restore(corpus, vocabulary, config, changed, state)
    save_checkpoint(state, tmp_path / "checkpoint")
    accumulator = tmp_path / "checkpoint" / "hidden_adagrad.npy"
    data = accumulator.read_bytes()
    accumulator.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_checkpoint(tmp_path / "checkpoint")
