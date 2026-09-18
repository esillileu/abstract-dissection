from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from w2v import (
    Corpus,
    Model,
    TrainingConfig,
    TrainingSession,
    Vocabulary,
    VocabularyConfig,
)

from f2.suites.w2v.artifacts import (
    CHECKPOINT_FORMAT,
    create_checkpoint_manager,
    load_checkpoint,
    load_lookup_artifact,
    save_checkpoint,
    save_lookup_artifact,
)
from repro_mlflow.artifact_cache import MlflowArtifactCache


def _session(path: Path, *, epochs: int = 2) -> tuple:
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
    return (
        corpus,
        vocabulary,
        config,
        TrainingSession(corpus, vocabulary, model, config),
    )


def test_checkpoint_roundtrip_resumes_bit_identically(tmp_path: Path) -> None:
    corpus, vocabulary, config, session = _session(tmp_path / "corpus.txt")
    session.train_epoch()
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(session.export_state(), checkpoint, resource_version="fixture-v1")
    loaded = load_checkpoint(
        checkpoint,
        expected={
            "config_digest": session.export_state().config_digest,
            "resource_version": "fixture-v1",
        },
    )
    resumed_model = Model.create(vocabulary, config)
    resumed = TrainingSession.restore(corpus, vocabulary, resumed_model, config, loaded)
    resumed.train_epoch()
    session.train_epoch()
    np.testing.assert_array_equal(
        resumed.export_state().input_embeddings(),
        session.export_state().input_embeddings(),
    )
    np.testing.assert_array_equal(
        resumed.export_state().output_embeddings(),
        session.export_state().output_embeddings(),
    )


def test_checkpoint_rejects_corruption_missing_files_and_identity(
    tmp_path: Path,
) -> None:
    *_, session = _session(tmp_path / "corrupt-corpus.txt", epochs=1)
    session.train_epoch()
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(session.export_state(), checkpoint, resource_version="fixture-v1")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_checkpoint(checkpoint, expected={"resource_version": "wrong"})

    (checkpoint / "counts.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_checkpoint(checkpoint)

    missing = tmp_path / "missing"
    save_checkpoint(session.export_state(), missing, resource_version="fixture-v1")
    (missing / "huffman_bits.npy").unlink()
    with pytest.raises(ValueError, match="file set"):
        load_checkpoint(missing)


def test_checkpoint_manager_pointers_and_periodic_retention(tmp_path: Path) -> None:
    *_, session = _session(tmp_path / "manager-corpus.txt", epochs=3)
    manager = create_checkpoint_manager(
        tmp_path / "managed",
        session=session,
        resource_version="fixture-v1",
    )
    session.train_epoch()
    latest = manager.save_latest()
    best = manager.save_best()
    session.train_epoch()
    final = manager.save_final()

    assert manager.current("latest") == latest
    assert manager.current("best") == best
    assert manager.current("final") == final
    assert load_checkpoint(final.path).completed_epochs == 2
    assert {path.name for path in (tmp_path / "managed").glob("*.json")} == {
        "latest.json",
        "best.json",
        "final.json",
    }


def test_lookup_is_mmap_backed_and_indexes_byte_tokens(tmp_path: Path) -> None:
    *_, session = _session(tmp_path / "lookup-corpus.txt", epochs=1)
    session.train_epoch()
    artifact = tmp_path / "embeddings"
    save_lookup_artifact(
        session.export_state(), artifact, resource_version="fixture-v1"
    )
    lookup = load_lookup_artifact(artifact)

    assert isinstance(lookup.embeddings, np.memmap)
    assert lookup.row(b"alpha") is not None
    np.testing.assert_array_equal(
        lookup.vector(b"alpha"), lookup.embeddings[lookup.row(b"alpha")]
    )
    assert lookup.vector(b"missing") is None


class _DownloadClient:
    tracking_uri = "https://mlflow.example.test"

    def __init__(self, source: Path) -> None:
        self.source = source

    def download_artifacts(
        self, run_id: str, artifact_path: str, destination: str
    ) -> str:
        assert run_id == "run-1"
        target = Path(destination) / Path(artifact_path).name
        shutil.copytree(self.source, target)
        return str(target)


def test_download_cache_roundtrip_preserves_verified_manifest(tmp_path: Path) -> None:
    *_, session = _session(tmp_path / "remote-corpus.txt", epochs=1)
    session.train_epoch()
    remote = tmp_path / "remote"
    save_checkpoint(session.export_state(), remote, resource_version="fixture-v1")
    cache = MlflowArtifactCache(
        _DownloadClient(remote),
        "https://mlflow.example.test",
        root=tmp_path / "cache",
    )
    downloaded = cache.get("run-1", "checkpoints/generation")
    restored = load_checkpoint(downloaded)

    assert restored.config_digest == session.export_state().config_digest
    assert (downloaded / "manifest.json").read_bytes() == (
        remote / "manifest.json"
    ).read_bytes()
    assert CHECKPOINT_FORMAT in (downloaded / "manifest.json").read_text()
