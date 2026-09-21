from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner
from w2v import (
    Corpus,
    Model,
    TrainingConfig,
    TrainingSession,
    Vocabulary,
    VocabularyConfig,
)

from f2.cli import app
from f2.suites.w2v import artifacts as artifact_module
from f2.suites.w2v.artifacts import (
    CHECKPOINT_FORMAT,
    create_checkpoint_manager,
    load_checkpoint,
    load_lookup_artifact,
    resolve_or_build_vocabulary,
    save_checkpoint,
    save_lookup_artifact,
)
from f2.suites.w2v.evaluate import evaluate_lookup_artifact
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
    save_checkpoint(session.export_state(), checkpoint)
    loaded = load_checkpoint(
        checkpoint,
        expected={
            "config_digest": session.export_state().config_digest,
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
    save_checkpoint(session.export_state(), checkpoint)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_checkpoint(checkpoint, expected={"config_digest": "wrong"})

    (checkpoint / "counts.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_checkpoint(checkpoint)

    missing = tmp_path / "missing"
    save_checkpoint(session.export_state(), missing)
    (missing / "huffman_bits.npy").unlink()
    with pytest.raises(ValueError, match="file set"):
        load_checkpoint(missing)


def test_checkpoint_manager_pointers_and_periodic_retention(tmp_path: Path) -> None:
    *_, session = _session(tmp_path / "manager-corpus.txt", epochs=3)
    manager = create_checkpoint_manager(
        tmp_path / "managed",
        session=session,
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
    save_lookup_artifact(session.export_state(), artifact)
    lookup = load_lookup_artifact(artifact)

    assert isinstance(lookup.embeddings, np.memmap)
    assert lookup.row(b"alpha") is not None
    np.testing.assert_array_equal(
        lookup.vector(b"alpha"), lookup.embeddings[lookup.row(b"alpha")]
    )
    assert lookup.vector(b"missing") is None


def test_shared_vocabulary_cache_reuses_verified_state_without_rebuilding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path = tmp_path / "vocabulary-corpus.txt"
    corpus_path.write_bytes(b"alpha beta alpha gamma\nbeta alpha delta\n")
    corpus = Corpus(corpus_path)
    values = {"initial_capacity": 2, "hash_capacity": 17, "min_count": 1}

    first = resolve_or_build_vocabulary(corpus, values, cache_root=tmp_path / "cache")
    original_build = Vocabulary.build
    calls = 0

    def fail_if_built(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(Vocabulary, "build", fail_if_built)
    second = resolve_or_build_vocabulary(corpus, values, cache_root=tmp_path / "cache")

    assert calls == 0
    assert second.digest() == first.digest()
    for actual, expected in zip(
        second.export_state().arrays(), first.export_state().arrays(), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_shared_vocabulary_cache_hit_reuses_supplied_corpus_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path = tmp_path / "verified-corpus.txt"
    corpus_path.write_bytes(b"alpha beta alpha\n")
    corpus = Corpus(corpus_path)
    digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    values = {"initial_capacity": 2, "hash_capacity": 17, "min_count": 1}
    resolve_or_build_vocabulary(
        corpus, values, cache_root=tmp_path / "cache", corpus_digest=digest
    )

    monkeypatch.setattr(
        Corpus, "digest", lambda _self: pytest.fail("cache hit rehashed corpus")
    )
    resolve_or_build_vocabulary(
        corpus, values, cache_root=tmp_path / "cache", corpus_digest=digest
    )


def test_concurrent_vocabulary_writer_adopts_valid_published_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path = tmp_path / "race-corpus.txt"
    corpus_path.write_bytes(b"alpha beta alpha\n")
    corpus = Corpus(corpus_path)
    values = {"initial_capacity": 2, "hash_capacity": 17, "min_count": 1}
    digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    cached = resolve_or_build_vocabulary(
        corpus, values, cache_root=tmp_path / "cache", corpus_digest=digest
    )
    target = next((tmp_path / "cache/f2/w2v/vocabulary").iterdir())
    semantic_config = {
        "min_count": 1,
        "max_lexical_words": 0,
        "hash_capacity": 17,
        "semantics_version": 1,
    }
    config_digest = artifact_module._json_digest(semantic_config)
    identity_digest = artifact_module._json_digest(
        {"corpus_digest": digest, "vocabulary_config_digest": config_digest}
    )

    def existing_target(_source: Path, destination: Path) -> None:
        if destination == target:
            raise FileExistsError(destination)
        raise AssertionError("unexpected publish destination")

    monkeypatch.setattr(artifact_module.os, "replace", existing_target)
    artifact_module._write_vocabulary_artifact(
        target,
        cached.export_state(),
        corpus_digest=digest,
        vocabulary_config=semantic_config,
        vocabulary_config_digest=config_digest,
        identity_digest=identity_digest,
    )
    assert (
        artifact_module._load_vocabulary_artifact(
            target,
            expected={"corpus_digest": digest, "vocabulary_config": semantic_config},
        ).digest()
        == cached.digest()
    )


@pytest.mark.parametrize(
    "change",
    (
        {"min_count": 2},
        {"max_lexical_words": 2},
        {"hash_capacity": 19},
    ),
)
def test_vocabulary_semantic_config_changes_cache_identity(
    tmp_path: Path, change: dict[str, int]
) -> None:
    corpus_path = tmp_path / "identity-corpus.txt"
    corpus_path.write_bytes(b"alpha beta alpha gamma\nbeta alpha delta\n")
    corpus = Corpus(corpus_path)
    base = {"initial_capacity": 2, "hash_capacity": 17, "min_count": 1}
    resolve_or_build_vocabulary(corpus, base, cache_root=tmp_path / "cache")
    changed = {**base, **change}
    resolve_or_build_vocabulary(corpus, changed, cache_root=tmp_path / "cache")

    artifacts = list((tmp_path / "cache/f2/w2v/vocabulary").iterdir())
    assert len(artifacts) == 2


def test_corrupt_shared_vocabulary_is_rebuilt_and_huffman_state_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path = tmp_path / "corrupt-vocabulary-corpus.txt"
    corpus_path.write_bytes(b"alpha beta alpha gamma\nbeta alpha delta\n")
    corpus = Corpus(corpus_path)
    values = {"initial_capacity": 2, "hash_capacity": 17, "min_count": 1}
    cache = tmp_path / "cache"
    fresh = Vocabulary.build(corpus, VocabularyConfig(**values))
    cached = resolve_or_build_vocabulary(corpus, values, cache_root=cache)
    artifact = next((cache / "f2/w2v/vocabulary").iterdir())
    (artifact / "huffman_bits.npy").write_bytes(b"corrupt")

    original_build = Vocabulary.build
    calls = 0

    def counted_build(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(Vocabulary, "build", counted_build)
    rebuilt = resolve_or_build_vocabulary(corpus, values, cache_root=cache)

    assert calls == 1
    assert rebuilt.digest() == fresh.digest() == cached.digest()
    for actual, expected in zip(
        rebuilt.export_state().arrays(), fresh.export_state().arrays(), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_saved_lookup_can_be_evaluated_without_training_or_overwrite(
    tmp_path: Path,
) -> None:
    *_, session = _session(tmp_path / "evaluation-corpus.txt", epochs=1)
    session.train_epoch()
    artifact = tmp_path / "lookup"
    save_lookup_artifact(session.export_state(), artifact)
    questions = tmp_path / "questions.txt"
    questions.write_bytes(b": relation\nalpha beta gamma delta\n")
    output = tmp_path / "reports" / "evaluation.json"

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "w2v1",
            "--lookup",
            str(artifact),
            "--questions",
            str(questions),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0
    report = json.loads(output.read_text())
    assert report["suite"] == "w2v1"
    assert report["evaluation_identity"] == {
        "questions_sha256": hashlib.sha256(questions.read_bytes()).hexdigest(),
        "phrase_separator": "_",
        "vocabulary_limit": 30_000,
    }
    assert report["analogy"]["overall"]["total_count"] == 1
    assert (artifact / "manifest.json").is_file()

    other_questions = tmp_path / "other-questions.txt"
    other_questions.write_bytes(b": relation\nalpha gamma beta delta\n")
    other_report = evaluate_lookup_artifact("w2v1", artifact, other_questions)
    assert (
        other_report["evaluation_identity"]["questions_sha256"]
        != report["evaluation_identity"]["questions_sha256"]
    )

    repeated = CliRunner().invoke(
        app,
        [
            "evaluate",
            "w2v1",
            "--lookup",
            str(artifact),
            "--questions",
            str(questions),
            "--output",
            str(output),
        ],
    )
    assert repeated.exit_code == 2
    assert "evaluation report already exists" in repeated.output


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
    save_checkpoint(session.export_state(), remote)
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
