from pathlib import Path

import numpy as np
from w2v import (
    Corpus,
    Model,
    TrainingConfig,
    TrainingSession,
    Vocabulary,
    VocabularyConfig,
)


def _objects(path: Path, *, epochs: int = 2):
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
