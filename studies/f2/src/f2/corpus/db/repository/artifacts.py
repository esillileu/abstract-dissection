"""Artifact, shard, and corpus statistics persistence operations."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseCorpusRepository


class ArtifactsRepositoryMixin(BaseCorpusRepository):
    """Repository operations for immutable artifacts, shards, and corpus stats."""

    def register_artifact(
        self,
        artifact_id: str,
        stage: str,
        s3_uri: str,
        sha256: str,
        byte_size: int,
        format: str,
        record_count: int | None = None,
        resource_version_id: str | None = None,
        acquisition_run_id: str | None = None,
        integrity_status: str = "pending",
        verification_report: dict[str, Any] | None = None,
    ) -> str:
        """Register an immutable artifact stored in SeaweedFS S3."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, stage, s3_uri, sha256, byte_size, format,
                    record_count, resource_version_id, acquisition_run_id,
                    integrity_status, verification_report
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (artifact_id) DO UPDATE SET
                    stage = EXCLUDED.stage,
                    s3_uri = EXCLUDED.s3_uri,
                    sha256 = EXCLUDED.sha256,
                    byte_size = EXCLUDED.byte_size,
                    format = EXCLUDED.format,
                    record_count = EXCLUDED.record_count,
                    integrity_status = EXCLUDED.integrity_status,
                    verification_report = EXCLUDED.verification_report;
                """,
                (
                    artifact_id,
                    stage,
                    s3_uri,
                    sha256,
                    byte_size,
                    format,
                    record_count,
                    resource_version_id,
                    acquisition_run_id,
                    integrity_status,
                    json.dumps(verification_report or {}),
                ),
            )
        self._commit()
        return artifact_id

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        """Retrieve artifact metadata by ID."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM artifacts WHERE artifact_id = %s;", (artifact_id,)
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row, strict=False))

    def update_artifact_verification(
        self,
        artifact_id: str,
        integrity_status: str,
        verification_report: dict[str, Any] | None = None,
    ) -> None:
        """Update integrity verification outcome for an artifact."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE artifacts
                SET integrity_status = %s,
                    verified_at = NOW(),
                    verification_report = %s
                WHERE artifact_id = %s;
                """,
                (integrity_status, json.dumps(verification_report or {}), artifact_id),
            )
        self._commit()

    def register_corpus_shard(
        self,
        resource_version_id: str,
        shard_index: int,
        artifact_id: str,
        word_count: int = 0,
        doc_count: int = 0,
        byte_size: int = 0,
    ) -> None:
        """Bind an artifact to an ordered corpus shard of a published version."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO corpus_shards (
                    resource_version_id, shard_index, artifact_id, word_count, doc_count, byte_size
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (resource_version_id, shard_index) DO UPDATE SET
                    artifact_id = EXCLUDED.artifact_id,
                    word_count = EXCLUDED.word_count,
                    doc_count = EXCLUDED.doc_count,
                    byte_size = EXCLUDED.byte_size;
                """,
                (
                    resource_version_id,
                    shard_index,
                    artifact_id,
                    word_count,
                    doc_count,
                    byte_size,
                ),
            )
        self._commit()

    def upsert_corpus_version_stats(
        self,
        resource_version_id: str,
        total_words: int,
        total_tokens: int,
        total_documents: int,
        total_sentences: int,
        total_bytes: int,
        total_shards: int,
        source_composition: dict[str, Any] | None = None,
        year_distribution: dict[str, Any] | None = None,
    ) -> None:
        """Upsert aggregated NLP corpus statistics."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO corpus_version_stats (
                    resource_version_id, total_words, total_tokens, total_documents,
                    total_sentences, total_bytes, total_shards, source_composition,
                    year_distribution, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (resource_version_id) DO UPDATE SET
                    total_words = EXCLUDED.total_words,
                    total_tokens = EXCLUDED.total_tokens,
                    total_documents = EXCLUDED.total_documents,
                    total_sentences = EXCLUDED.total_sentences,
                    total_bytes = EXCLUDED.total_bytes,
                    total_shards = EXCLUDED.total_shards,
                    source_composition = EXCLUDED.source_composition,
                    year_distribution = EXCLUDED.year_distribution,
                    updated_at = NOW();
                """,
                (
                    resource_version_id,
                    total_words,
                    total_tokens,
                    total_documents,
                    total_sentences,
                    total_bytes,
                    total_shards,
                    json.dumps(source_composition or {}),
                    json.dumps(year_distribution or {}),
                ),
            )
        self._commit()

    def get_corpus_version_stats(
        self, resource_version_id: str
    ) -> dict[str, Any] | None:
        """Retrieve corpus version statistics."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM corpus_version_stats WHERE resource_version_id = %s;",
                (resource_version_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row, strict=False))

    def list_verified_corpus_shards(
        self, resource_version_id: str
    ) -> list[dict[str, Any]]:
        """Return the immutable ordered shard identity used by training adapters."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.shard_index, a.s3_uri, a.sha256, s.byte_size,
                       s.word_count, s.doc_count
                FROM corpus_shards AS s
                JOIN artifacts AS a ON a.artifact_id = s.artifact_id
                WHERE s.resource_version_id = %s
                  AND a.resource_version_id = s.resource_version_id
                  AND a.integrity_status = 'verified'
                ORDER BY s.shard_index;
                """,
                (resource_version_id,),
            )
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]


__all__ = ["ArtifactsRepositoryMixin"]
