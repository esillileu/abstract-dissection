"""Unit tests verifying InMemoryCorpusRepository and CorpusStateRepositoryProtocol."""

from __future__ import annotations

from f2.corpus.db.repository import (
    CorpusStateRepositoryProtocol,
    InMemoryCorpusRepository,
)
from f2.corpus.lifecycle.catalog import install_validation_profiles


def test_in_memory_repository_implements_protocol():
    repo = InMemoryCorpusRepository()
    assert isinstance(repo, CorpusStateRepositoryProtocol)


def test_in_memory_repository_lifecycle(in_memory_repository):
    repo = in_memory_repository

    # 1. Validation profile catalog installation
    install_validation_profiles(repo)
    raw_profile = repo.get_validation_profile("raw-integrity-v1")
    assert raw_profile is not None
    assert raw_profile["target_paper_id"] is None
    assert raw_profile["is_active"] is True

    # 2. Acquisition run lifecycle
    run_id = repo.create_acquisition_run(
        run_id="acq_001",
        resource_version_id="wmt_v1",
        acquisition_method="download",
        code_version="1.0.0",
        config_hash="abc123hash",
    )
    assert repo.get_acquisition_run(run_id)["status"] == "running"
    repo.finish_acquisition_run(
        run_id, status="completed", stats={"downloaded_bytes": 1024}
    )
    assert repo.get_acquisition_run(run_id)["status"] == "completed"

    # 3. Artifact registration and verification
    art_id = repo.register_artifact(
        artifact_id="art_001",
        stage="raw",
        s3_uri="s3://corpus/raw/doc.txt",
        sha256="deadbeef" * 8,
        byte_size=1024,
        format="text",
    )
    assert repo.get_artifact(art_id)["integrity_status"] == "pending"
    repo.update_artifact_verification(art_id, integrity_status="verified")
    assert repo.get_artifact(art_id)["integrity_status"] == "verified"

    # 4. Corpus shard and stats
    repo.register_corpus_shard(
        shard_id="shard_001",
        resource_version_id="wmt_v1",
        artifact_id=art_id,
        shard_index=0,
        total_shards=1,
        token_count=250,
        byte_size=1024,
    )
    shards = repo.list_verified_corpus_shards("wmt_v1")
    assert len(shards) == 1
    assert shards[0]["token_count"] == 250

    repo.upsert_corpus_version_stats(
        resource_version_id="wmt_v1",
        total_tokens=250,
        total_bytes=1024,
        total_shards=1,
    )
    stats = repo.get_corpus_version_stats("wmt_v1")
    assert stats["total_tokens"] == 250

    # 5. Validation run recording
    val_id = repo.record_validation_run(
        validation_run_id="val_001",
        artifact_id=art_id,
        profile_id="raw-integrity-v1",
        status="passed",
    )
    assert repo.get_validation_run(val_id)["status"] == "passed"
    history = repo.get_validation_history(art_id)
    assert len(history) == 1
    assert history[0]["validation_run_id"] == "val_001"
