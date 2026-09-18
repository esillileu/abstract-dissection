from pathlib import Path

import numpy as np
import pytest
from w2v import (
    Corpus,
    Model,
    TrainingConfig,
    TrainingSession,
    TrainingState,
    Vocabulary,
    VocabularyConfig,
    VocabularyState,
)


def _objects(path: Path, *, epochs: int = 2, observation_interval: int = 0):
    path.write_bytes(b"alpha beta alpha gamma\nbeta alpha delta\ngamma beta alpha\n")
    corpus = Corpus(path)
    vocabulary = Vocabulary.build(
        corpus,
        VocabularyConfig(initial_capacity=2, hash_capacity=17, min_count=1),
    )
    config = TrainingConfig(
        model_kind="skip_gram",
        embedding_dimension=4,
        window_radius=2,
        epochs=epochs,
        thread_count=1,
        subsampling_threshold=0.0,
        negative_sample_count=2,
        negative_table_size=257,
        sigmoid_table_size=101,
        observation_interval=observation_interval,
    )
    model = Model.create(vocabulary, config)
    return corpus, vocabulary, config, model


def test_vocabulary_and_embedding_numpy_roundtrip(tmp_path: Path) -> None:
    _, vocabulary, _, model = _objects(tmp_path / "corpus.txt")

    token_bytes = vocabulary.token_bytes()
    offsets = vocabulary.token_offsets()
    counts = vocabulary.counts()
    assert token_bytes.dtype == np.uint8
    assert offsets.dtype == np.uint64
    assert counts.dtype == np.uint64
    assert offsets.shape == (vocabulary.size + 1,)
    assert offsets[-1] == token_bytes.size

    initial = model.input_embeddings()
    assert initial.dtype == np.float32
    assert initial.shape == (vocabulary.size, 4)
    replacement = np.arange(initial.size, dtype=np.float32).reshape(initial.shape)
    model.restore_input_embeddings(replacement)
    np.testing.assert_array_equal(model.input_embeddings(), replacement)


def test_vocabulary_state_includes_huffman_arrays(tmp_path: Path) -> None:
    _, vocabulary, _, _ = _objects(tmp_path / "vocabulary.txt")
    state = vocabulary.export_state()
    arrays = state.arrays()
    token_bytes, token_offsets, counts, huffman_offsets, paths, bits = arrays

    assert token_offsets.shape == (vocabulary.size + 1,)
    assert huffman_offsets.shape == (vocabulary.size + 1,)
    assert huffman_offsets[-1] == paths.size == bits.size
    restored_state = VocabularyState.from_arrays(
        token_bytes,
        token_offsets,
        counts,
        huffman_offsets,
        paths,
        bits,
        state.hash_capacity,
        state.retained_token_count,
    )
    restored = Vocabulary.restore_state(restored_state)
    assert restored.digest() == vocabulary.digest() == state.digest()
    restored_arrays = restored.export_state().arrays()
    for expected, actual in zip(arrays, restored_arrays, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_vocabulary_state_rejects_malformed_arrays(tmp_path: Path) -> None:
    _, vocabulary, _, _ = _objects(tmp_path / "malformed-vocabulary.txt")
    state = vocabulary.export_state()
    token_bytes, token_offsets, counts, huffman_offsets, paths, bits = state.arrays()

    bad_offsets = token_offsets.copy()
    bad_offsets[1] = token_bytes.size + 1
    with pytest.raises(RuntimeError, match="corrupt data"):
        VocabularyState.from_arrays(
            token_bytes,
            bad_offsets,
            counts,
            huffman_offsets,
            paths,
            bits,
            state.hash_capacity,
            state.retained_token_count,
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        VocabularyState.from_arrays(
            token_bytes[::-1],
            token_offsets,
            counts,
            huffman_offsets,
            paths,
            bits,
            state.hash_capacity,
            state.retained_token_count,
        )


def test_embedding_restore_rejects_invalid_arrays_atomically(tmp_path: Path) -> None:
    _, vocabulary, _, model = _objects(tmp_path / "invalid-embedding.txt")
    original = model.input_embeddings()
    shape = original.shape

    with pytest.raises(ValueError):
        model.restore_input_embeddings(np.zeros((shape[0], shape[1] + 1), np.float32))
    with pytest.raises(TypeError):
        model.restore_input_embeddings(np.zeros(shape, np.float64))
    with pytest.raises(ValueError, match="C-contiguous"):
        model.restore_input_embeddings(np.zeros(shape[::-1], np.float32).T)
    non_finite = original.copy()
    non_finite[0, 0] = np.nan
    with pytest.raises(ValueError, match="invalid state"):
        model.restore_input_embeddings(non_finite)
    np.testing.assert_array_equal(model.input_embeddings(), original)


def test_epoch_state_resume_matches_continuous_training(tmp_path: Path) -> None:
    corpus, vocabulary, config, checkpoint_model = _objects(tmp_path / "checkpoint.txt")
    checkpoint_session = TrainingSession(corpus, vocabulary, checkpoint_model, config)
    first_report = checkpoint_session.train_epoch()
    state = checkpoint_session.export_state()
    assert first_report.epoch == 1
    assert state.completed_epochs == 1
    np.testing.assert_array_equal(
        state.input_embeddings(), checkpoint_model.input_embeddings()
    )

    resumed_model = Model.create(vocabulary, config)
    resumed_session = TrainingSession.restore(
        corpus, vocabulary, resumed_model, config, state
    )
    second_report = resumed_session.train_epoch()
    assert second_report.epoch == 2
    assert resumed_session.is_complete

    continuous_model = Model.create(vocabulary, config)
    continuous_session = TrainingSession(corpus, vocabulary, continuous_model, config)
    continuous_session.train_epoch()
    continuous_session.train_epoch()
    np.testing.assert_array_equal(
        resumed_model.input_embeddings(), continuous_model.input_embeddings()
    )
    np.testing.assert_array_equal(
        resumed_model.output_embeddings(), continuous_model.output_embeddings()
    )


def test_training_state_parts_roundtrip(tmp_path: Path) -> None:
    corpus, vocabulary, config, model = _objects(tmp_path / "parts.txt")
    session = TrainingSession(corpus, vocabulary, model, config)
    session.train_epoch()
    state = session.export_state()
    vocabulary_state = state.vocabulary_state()
    restored_state = TrainingState.from_parts(
        state.schema_version,
        state.config_digest,
        state.vocabulary_digest,
        state.corpus_digest,
        state.completed_epochs,
        state.processed_tokens,
        vocabulary_state,
        state.input_embeddings(),
        state.output_embeddings(),
        state.workers(),
    )
    restored_vocabulary = Vocabulary.restore_state(vocabulary_state)
    restored_model = Model.create(restored_vocabulary, config)
    resumed = TrainingSession.restore(
        corpus, restored_vocabulary, restored_model, config, restored_state
    )
    resumed.train_epoch()

    continuous_model = Model.create(vocabulary, config)
    continuous = TrainingSession.restore(
        corpus, vocabulary, continuous_model, config, state
    )
    continuous.train_epoch()
    np.testing.assert_array_equal(
        resumed.export_state().input_embeddings(),
        continuous.export_state().input_embeddings(),
    )


def test_callback_exception_leaves_completed_epoch_exportable(tmp_path: Path) -> None:
    corpus, vocabulary, config, model = _objects(tmp_path / "callback.txt")
    session = TrainingSession(corpus, vocabulary, model, config)

    def fail(report) -> None:
        assert report.epoch == 1
        raise LookupError("callback failed")

    with pytest.raises(LookupError, match="callback failed"):
        session.train_epoch(fail)
    assert session.completed_epochs == 1
    assert session.export_state().completed_epochs == 1
    assert session.train_epoch().epoch == 2
    assert session.is_complete


def test_dense_objective_observations_are_exposed_to_python(tmp_path: Path) -> None:
    corpus, vocabulary, config, model = _objects(
        tmp_path / "observations.txt", epochs=1, observation_interval=1
    )
    report = TrainingSession(corpus, vocabulary, model, config).train_epoch()
    observations = report.observations()

    assert observations
    assert report.objective_loss_count == sum(
        observation.objective_loss_count for observation in observations
    )
    assert report.objective_loss == pytest.approx(
        report.objective_loss_sum / report.objective_loss_count
    )
    assert all(
        np.isfinite(observation.objective_loss)
        and observation.epoch == 1
        and observation.processed_tokens <= report.processed_tokens
        for observation in observations
    )
