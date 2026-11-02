"""In-memory stub repository for isolated offline and unit testing without PostgreSQL."""

from __future__ import annotations

import datetime
from typing import Any


class InMemoryCorpusRepository:
    """In-memory state repository replacing PostgreSQL for fast, disposable testing."""

    def __init__(self) -> None:
        self.acquisition_runs: dict[str, dict[str, Any]] = {}
        self.artifacts: dict[str, dict[str, Any]] = {}
        self.corpus_shards: dict[str, dict[str, Any]] = {}
        self.corpus_stats: dict[str, dict[str, Any]] = {}
        self.processing_runs: dict[str, dict[str, Any]] = {}
        self.processing_io: list[dict[str, Any]] = []
        self.validation_profiles: dict[str, dict[str, Any]] = {}
        self.validation_runs: dict[str, dict[str, Any]] = {}

    def create_acquisition_run(
        self,
        run_id: str,
        resource_version_id: str,
        acquisition_method: str,
        code_version: str,
        config_hash: str,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self.acquisition_runs[run_id] = {
            "run_id": run_id,
            "resource_version_id": resource_version_id,
            "acquisition_method": acquisition_method,
            "code_version": code_version,
            "config_hash": config_hash,
            "config": config or {},
            "status": "running",
            "metadata": metadata or {},
            "created_at": datetime.datetime.now(datetime.UTC),
            "finished_at": None,
            "error_message": None,
            "stats": None,
        }
        return run_id

    def finish_acquisition_run(
        self,
        run_id: str,
        status: str,
        error_message: str | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        if run_id in self.acquisition_runs:
            self.acquisition_runs[run_id].update(
                status=status,
                finished_at=datetime.datetime.now(datetime.UTC),
                error_message=error_message,
                stats=stats,
            )

    def get_acquisition_run(self, run_id: str) -> dict[str, Any] | None:
        return self.acquisition_runs.get(run_id)

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
        self.artifacts[artifact_id] = {
            "artifact_id": artifact_id,
            "stage": stage,
            "s3_uri": s3_uri,
            "sha256": sha256,
            "byte_size": byte_size,
            "format": format,
            "record_count": record_count,
            "resource_version_id": resource_version_id,
            "acquisition_run_id": acquisition_run_id,
            "integrity_status": integrity_status,
            "verification_report": verification_report or {},
            "created_at": datetime.datetime.now(datetime.UTC),
        }
        return artifact_id

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        return self.artifacts.get(artifact_id)

    def update_artifact_verification(
        self,
        artifact_id: str,
        integrity_status: str,
        verification_report: dict[str, Any] | None = None,
    ) -> None:
        if artifact_id in self.artifacts:
            self.artifacts[artifact_id]["integrity_status"] = integrity_status
            if verification_report is not None:
                self.artifacts[artifact_id]["verification_report"] = verification_report

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
    ) -> str:
        self.corpus_shards[shard_id] = {
            "shard_id": shard_id,
            "resource_version_id": resource_version_id,
            "artifact_id": artifact_id,
            "shard_index": shard_index,
            "total_shards": total_shards,
            "token_count": token_count,
            "byte_size": byte_size,
            "line_count": line_count,
            "format": format,
        }
        return shard_id

    def upsert_corpus_version_stats(
        self,
        resource_version_id: str,
        total_tokens: int,
        total_bytes: int,
        total_shards: int,
        total_records: int | None = None,
        unigram_counts_json: dict[str, Any] | None = None,
        phrase_counts_json: dict[str, Any] | None = None,
    ) -> None:
        self.corpus_stats[resource_version_id] = {
            "resource_version_id": resource_version_id,
            "total_tokens": total_tokens,
            "total_bytes": total_bytes,
            "total_shards": total_shards,
            "total_records": total_records,
            "unigram_counts_json": unigram_counts_json or {},
            "phrase_counts_json": phrase_counts_json or {},
        }

    def get_corpus_version_stats(
        self, resource_version_id: str
    ) -> dict[str, Any] | None:
        return self.corpus_stats.get(resource_version_id)

    def list_verified_corpus_shards(
        self, resource_version_id: str
    ) -> list[dict[str, Any]]:
        shards = [
            s
            for s in self.corpus_shards.values()
            if s["resource_version_id"] == resource_version_id
        ]
        shards.sort(key=lambda s: s["shard_index"])
        return shards

    def create_processing_run(
        self,
        run_id: str,
        recipe_name: str,
        recipe_version: str,
        code_version: str,
        config_hash: str,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self.processing_runs[run_id] = {
            "run_id": run_id,
            "recipe_name": recipe_name,
            "recipe_version": recipe_version,
            "code_version": code_version,
            "config_hash": config_hash,
            "config": config or {},
            "status": "running",
            "metadata": metadata or {},
            "created_at": datetime.datetime.now(datetime.UTC),
            "finished_at": None,
            "error_message": None,
            "stats": None,
        }
        return run_id

    def finish_processing_run(
        self,
        run_id: str,
        status: str,
        error_message: str | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        if run_id in self.processing_runs:
            self.processing_runs[run_id].update(
                status=status,
                finished_at=datetime.datetime.now(datetime.UTC),
                error_message=error_message,
                stats=stats,
            )

    def get_processing_run(self, run_id: str) -> dict[str, Any] | None:
        return self.processing_runs.get(run_id)

    def record_processing_io(
        self,
        run_id: str,
        input_artifact_ids: list[str],
        output_artifact_ids: list[str],
    ) -> None:
        self.processing_io.append(
            {
                "run_id": run_id,
                "input_artifact_ids": list(input_artifact_ids),
                "output_artifact_ids": list(output_artifact_ids),
            }
        )

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
    ) -> str:
        self.validation_profiles[profile_id] = {
            "profile_id": profile_id,
            "profile_key": profile_key,
            "revision": revision,
            "name": name,
            "spec_hash": spec_hash,
            "specification": specification,
            "description": description,
            "target_paper_id": target_paper_id,
            "notes": notes,
            "is_active": True,
            "created_at": datetime.datetime.now(datetime.UTC),
        }
        return profile_id

    def deprecate_validation_profile(
        self, profile_id: str, notes: str | None = None
    ) -> None:
        if profile_id in self.validation_profiles:
            self.validation_profiles[profile_id]["is_active"] = False
            if notes is not None:
                self.validation_profiles[profile_id]["notes"] = notes

    def get_validation_profile(self, profile_id: str) -> dict[str, Any] | None:
        return self.validation_profiles.get(profile_id)

    def record_validation_run(
        self,
        validation_run_id: str,
        artifact_id: str,
        profile_id: str,
        status: str,
        details: dict[str, Any] | None = None,
        execution_time_ms: int | None = None,
    ) -> str:
        self.validation_runs[validation_run_id] = {
            "validation_run_id": validation_run_id,
            "artifact_id": artifact_id,
            "profile_id": profile_id,
            "status": status,
            "details": details or {},
            "execution_time_ms": execution_time_ms,
            "created_at": datetime.datetime.now(datetime.UTC),
        }
        return validation_run_id

    def get_validation_run(self, validation_run_id: str) -> dict[str, Any] | None:
        return self.validation_runs.get(validation_run_id)

    def get_validation_history(
        self, artifact_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        runs = [
            r for r in self.validation_runs.values() if r["artifact_id"] == artifact_id
        ]
        runs.sort(key=lambda r: r["created_at"], reverse=True)
        return runs[:limit]
