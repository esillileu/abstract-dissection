"""Source raw file download, verification, and S3 registration."""

from __future__ import annotations

from pathlib import Path

from repro_io.checksum import sha256_file
from repro_io.http.download import BandwidthScheduler, SerialDownloader
from repro_io.s3 import S3ObjectStore

from ..db.repository import CorpusStateRepository
from ..sources import CorpusSource, stable_id
from .preflight import git_sha


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
    downloader = SerialDownloader(
        user_agent="abstract-dissection-f2/1.0", bandwidth=bandwidth
    )
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


__all__ = ["acquire_source"]
