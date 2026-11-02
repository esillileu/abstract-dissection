"""Repository protocol defining the abstract contract for corpus state operations."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CorpusStateRepositoryProtocol(Protocol):
    """Protocol defining the corpus state repository interface."""

    def create_acquisition_run(
        self,
        run_id: str,
        resource_version_id: str,
        acquisition_method: str,
        code_version: str,
        config_hash: str,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str: ...

    def finish_acquisition_run(
        self,
        run_id: str,
        status: str,
        error_message: str | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None: ...

    def get_acquisition_run(self, run_id: str) -> dict[str, Any] | None: ...

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
    ) -> str: ...

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None: ...

    def update_artifact_verification(
        self,
        artifact_id: str,
        integrity_status: str,
        verification_report: dict[str, Any] | None = None,
    ) -> None: ...

    def register_corpus_shard(
        self,
        shard_id: str,
        resource_version_id: str,
        artifact_id: str,
        shard_index: int,
        total_shards: int,
        token_count: int,
        byte_size: int,
        line_count: int | None = None,
        format: str = "jsonl.zst",
    ) -> str: ...

    def upsert_corpus_version_stats(
        self,
        resource_version_id: str,
        total_tokens: int,
        total_bytes: int,
        total_shards: int,
        total_records: int | None = None,
        unigram_counts_json: dict[str, Any] | None = None,
        phrase_counts_json: dict[str, Any] | None = None,
    ) -> None: ...

    def get_corpus_version_stats(
        self, resource_version_id: str
    ) -> dict[str, Any] | None: ...

    def list_verified_corpus_shards(
        self, resource_version_id: str
    ) -> list[dict[str, Any]]: ...

    def create_processing_run(
        self,
        run_id: str,
        recipe_name: str,
        recipe_version: str,
        code_version: str,
        config_hash: str,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str: ...

    def finish_processing_run(
        self,
        run_id: str,
        status: str,
        error_message: str | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None: ...

    def get_processing_run(self, run_id: str) -> dict[str, Any] | None: ...

    def record_processing_io(
        self,
        run_id: str,
        input_artifact_ids: list[str],
        output_artifact_ids: list[str],
    ) -> None: ...

    def create_validation_profile(
        self,
        profile_id: str,
        profile_key: str,
        revision: int,
        name: str,
        spec_hash: str,
        specification: dict[str, Any],
        description: str | None = None,
        target_paper_id: str | None = None,
        notes: str | None = None,
    ) -> str: ...

    def deprecate_validation_profile(
        self, profile_id: str, notes: str | None = None
    ) -> None: ...

    def get_validation_profile(self, profile_id: str) -> dict[str, Any] | None: ...

    def record_validation_run(
        self,
        validation_run_id: str,
        artifact_id: str,
        profile_id: str,
        status: str,
        details: dict[str, Any] | None = None,
        execution_time_ms: int | None = None,
    ) -> str: ...

    def get_validation_run(self, validation_run_id: str) -> dict[str, Any] | None: ...

    def get_validation_history(
        self, artifact_id: str, limit: int = 10
    ) -> list[dict[str, Any]]: ...
