"""Canonical non-Common-Crawl corpus lifecycle orchestration."""

from __future__ import annotations

import hashlib
import json
import locale
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .canonical import (
    BandwidthScheduler,
    DeterministicSharder,
    SerialDownloader,
    ShardInfo,
    normalize_text,
    open_canonical_shards,
    open_source_records,
    sha256_file,
)
from .db.repository import CorpusStateRepository
from .object_store import S3ObjectStore
from .sources import (
    SOURCE_BOUNDARY_POLICIES,
    SOURCES,
    VALIDATION_PROFILES,
    CorpusSource,
    stable_id,
)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def config_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def catalog_sources(conn: Any) -> None:
    """Idempotently register source, canonical, and normalized resource versions."""
    with conn.cursor() as cur:
        for source in SOURCES:
            status = "blocked" if source.blocked_reason else "pending"
            for suffix, name, access in (
                ("raw", f"{source.name} raw release", source.access),
                ("canonical", f"{source.name} canonical text", source.access),
                (
                    "normalized",
                    f"{source.name} word2vec-normalized text",
                    source.access,
                ),
            ):
                resource_id = f"f2-{source.key}-{suffix}"
                version_id = getattr(source, f"{suffix}_resource_version_id")
                cur.execute(
                    """INSERT INTO catalog.resources
                       (resource_id, kind, name, description, access_status, acquisition_status, readiness_status, notes)
                       VALUES (%s, 'dataset', %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (resource_id) DO UPDATE SET name=EXCLUDED.name, description=EXCLUDED.description,
                         access_status=EXCLUDED.access_status, acquisition_status=EXCLUDED.acquisition_status,
                         readiness_status=EXCLUDED.readiness_status, notes=EXCLUDED.notes""",
                    (
                        resource_id,
                        name,
                        f"Frozen F2 source {source.key} release {source.release}",
                        access,
                        status,
                        status,
                        source.blocked_reason,
                    ),
                )
                metadata = {
                    "source_key": source.key,
                    "representation": suffix,
                    "release": source.release,
                    "source_config_hash": source.config_hash,
                }
                cur.execute(
                    """INSERT INTO catalog.resource_versions
                       (resource_version_id, resource_id, version_label, uri, metadata, is_verified)
                       VALUES (%s, %s, %s, %s, %s, FALSE)
                       ON CONFLICT (resource_version_id) DO NOTHING""",
                    (
                        version_id,
                        resource_id,
                        source.release if suffix == "raw" else "v1",
                        source.homepage if suffix == "raw" else None,
                        json.dumps(metadata),
                    ),
                )
            cur.execute(
                """INSERT INTO catalog.resource_sources (resource_id, source_type, url, license, is_preferred, notes)
                   SELECT %s, 'official_release', %s, %s, TRUE, %s
                   WHERE NOT EXISTS (SELECT 1 FROM catalog.resource_sources WHERE resource_id=%s AND is_preferred=TRUE)""",
                (
                    f"f2-{source.key}-raw",
                    source.homepage,
                    source.license,
                    source.blocked_reason,
                    f"f2-{source.key}-raw",
                ),
            )
    conn.commit()


def install_validation_profiles(repo: CorpusStateRepository) -> None:
    for key, spec in VALIDATION_PROFILES.items():
        profile_id = key
        if repo.get_validation_profile(profile_id):
            continue
        repo.create_validation_profile(
            profile_id,
            key.rsplit("-v", 1)[0],
            spec["revision"],
            key,
            config_hash(spec),
            spec,
        )


def acquire_source(
    source: CorpusSource,
    staging: Path,
    store: S3ObjectStore,
    repo: CorpusStateRepository,
    *,
    checksum_overrides: dict[str, str] | None = None,
    bandwidth: BandwidthScheduler | None = None,
) -> list[str]:
    if source.blocked_reason:
        raise PermissionError(source.blocked_reason)
    checksum_overrides = checksum_overrides or {}
    run_id = "acq-" + stable_id(source.key, source.release, source.config_hash)
    existing = repo.get_acquisition_run(run_id)
    if existing and existing["status"] == "completed":
        return ["already-completed"]
    repo.create_acquisition_run(
        run_id,
        source.raw_resource_version_id,
        "http_archive_download",
        git_sha(),
        source.config_hash,
        {"source": source.key, "release": source.release},
    )
    output: list[str] = []
    downloader = SerialDownloader(bandwidth=bandwidth)
    try:
        for item in source.files:
            expected = checksum_overrides.get(item.name) or item.sha256
            if source.key in {"lm1b", "wikipedia"} and not expected:
                raise ValueError(
                    f"{source.key} requires an upstream SHA-256 for {item.name}"
                )
            local = downloader.download(
                item.url,
                staging / source.key / "raw" / item.name,
                expected_sha256=expected,
            )
            digest = sha256_file(local)
            artifact_id = "raw-" + stable_id(
                source.key, source.release, item.name, digest
            )
            uri = store.uri(f"raw/{source.key}/{source.release}/{item.name}")
            store.put_file(local, uri)
            remote_digest = store.sha256(uri)
            if remote_digest != digest:
                raise OSError(f"post-upload SHA-256 mismatch for {item.name}")
            with repo.conn.transaction():
                repo.register_artifact(
                    artifact_id,
                    "raw",
                    uri,
                    digest,
                    local.stat().st_size,
                    "tar.gz"
                    if source.key in {"lm1b", "umbc"}
                    else item.name.rsplit(".", 1)[-1],
                    resource_version_id=source.raw_resource_version_id,
                    acquisition_run_id=run_id,
                    integrity_status="verified",
                    verification_report={
                        "upstream_sha256": expected,
                        "remote_sha256": remote_digest,
                        "year": item.year,
                    },
                )
            local.unlink()
            output.append(artifact_id)
        repo.finish_acquisition_run(
            run_id, "completed", metadata={"artifact_ids": output}
        )
        return output
    except Exception as exc:
        repo.finish_acquisition_run(run_id, "failed", error_message=str(exc))
        raise


def process_canonical_source(
    source: CorpusSource,
    raw_inputs: list[Path],
    output_dir: Path,
    *,
    target_words: int = 10_000_000,
) -> tuple[list[ShardInfo], Path]:
    """Extract visible/released text from raw archives without lowercase or token normalization."""
    records = open_source_records(source.key, raw_inputs)
    policy = SOURCE_BOUNDARY_POLICIES.get(source.key, "sentence_per_line")
    sharder = DeterministicSharder(output_dir, target_words)
    shards = sharder.write(records, source=source.key, boundary_policy=policy)
    return shards, output_dir / "manifest.json"


def process_normalized_source(
    source: CorpusSource,
    canonical_shard_paths: list[Path],
    output_dir: Path,
    *,
    target_words: int = 10_000_000,
) -> tuple[list[ShardInfo], Path]:
    """Apply Mikolov demo-train-big-model-v1.sh normalize_text() to canonical shards."""
    canonical_records = open_canonical_shards(canonical_shard_paths)
    records = ((rec_id, normalize_text(text)) for rec_id, text in canonical_records)
    policy = SOURCE_BOUNDARY_POLICIES.get(source.key, "sentence_per_line")
    sharder = DeterministicSharder(output_dir, target_words)
    shards = sharder.write(records, source=source.key, boundary_policy=policy)
    return shards, output_dir / "manifest.json"


def process_source(
    source: CorpusSource,
    inputs: list[Path],
    output_dir: Path,
    *,
    normalized: bool = True,
    target_words: int = 10_000_000,
) -> tuple[list[Any], Path]:
    if normalized:
        records = open_source_records(source.key, inputs)
        norm_records = ((ident, normalize_text(text)) for ident, text in records)
        policy = SOURCE_BOUNDARY_POLICIES.get(source.key, "sentence_per_line")
        sharder = DeterministicSharder(output_dir, target_words)
        shards = sharder.write(norm_records, source=source.key, boundary_policy=policy)
        return shards, output_dir / "manifest.json"
    return process_canonical_source(
        source, inputs, output_dir, target_words=target_words
    )


def tool_versions() -> dict[str, str]:
    versions: dict[str, str] = {"locale": locale.setlocale(locale.LC_ALL, None)}
    for tool, args in {
        "zstd": ["--version"],
        "awk": ["--version"],
        "sed": ["--version"],
        "perl": ["-v"],
    }.items():
        try:
            line = subprocess.check_output(
                [tool, *args], text=True, stderr=subprocess.STDOUT
            ).splitlines()[0]
        except (OSError, subprocess.CalledProcessError):
            line = "unavailable"
        versions[tool] = line
    return versions


def preflight(staging_root: Path, store: S3ObjectStore | None = None) -> dict[str, Any]:
    usage = shutil.disk_usage(
        staging_root if staging_root.exists() else staging_root.parent
    )
    result: dict[str, Any] = {
        "git_sha": git_sha(),
        "staging_root": staging_root.as_posix(),
        "scratch_free_bytes": usage.free,
        "scratch_187gb": usage.free >= 187_000_000_000,
        "tools": tool_versions(),
    }
    if store is not None:
        store.probe()
        result["s3"] = "reachable"
        result["s3_root"] = store.config.root_uri
    return result


__all__ = [
    "acquire_source",
    "catalog_sources",
    "config_hash",
    "git_sha",
    "install_validation_profiles",
    "preflight",
    "process_canonical_source",
    "process_normalized_source",
    "process_source",
    "tool_versions",
]
