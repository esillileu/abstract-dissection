"""Stage 1 processing orchestrator: raw inputs to source-canonical shards."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import psycopg
import typer
from repro_io.checksum import sha256_file
from repro_io.s3 import S3ObjectStore

from repro_core.context.paths import RuntimePaths

from ..db.repository import CorpusStateRepository
from ..lifecycle import (
    config_hash,
    git_sha,
    process_canonical_source,
    tool_versions,
)
from ..sources import CorpusSource, stable_id


def _raw_inputs(
    repo: CorpusStateRepository, spec: CorpusSource, store: S3ObjectStore, target: Path
) -> list[Path]:
    target.mkdir(parents=True, exist_ok=True)
    with repo.conn.cursor() as cur:
        cur.execute(
            "SELECT s3_uri, sha256 FROM artifacts WHERE resource_version_id=%s AND stage='raw' ORDER BY s3_uri",
            (spec.raw_resource_version_id,),
        )
        rows = cur.fetchall()
    result: list[Path] = []
    for uri, digest in rows:
        if not uri.startswith(store.config.root_uri):
            continue
        local = target / Path(uri).name
        if not local.exists() or sha256_file(local) != digest:
            store.get_file(uri, local)
        if sha256_file(local) != digest:
            raise OSError(f"downloaded raw object checksum mismatch: {uri}")
        result.append(local)
    return result


def _process_canonical_stage(
    conn: psycopg.Connection,
    repo: CorpusStateRepository,
    spec: CorpusSource,
    store: S3ObjectStore,
    paths: RuntimePaths,
    target_words: int,
    boundary_policy: str,
) -> None:
    raw_dir = paths.staging_root / "exp" / "f2" / "sources" / spec.key / "raw-cache"
    raw_inputs = _raw_inputs(repo, spec, store, raw_dir)
    if not raw_inputs:
        raise RuntimeError("no verified raw artifacts; run acquire first")

    canonical_config = {
        "source": spec.key,
        "stage": "source-canonical",
        "target_words": target_words,
        "boundary_policy": boundary_policy,
        "tools": tool_versions(),
    }
    canonical_digest = config_hash(canonical_config)
    canonical_run_id = "proc-" + stable_id(
        spec.canonical_resource_version_id, canonical_digest
    )
    previous_canonical = repo.get_processing_run(canonical_run_id)
    canonical_output_dir = (
        paths.staging_root / "exp" / "f2" / "sources" / spec.key / "canonical"
    )
    canonical_stats = repo.get_corpus_version_stats(spec.canonical_resource_version_id)

    if (
        previous_canonical
        and previous_canonical["status"] == "completed"
        and canonical_stats
    ):
        typer.echo(f"{spec.key}/canonical: already completed")
        return
    if canonical_stats and not previous_canonical:
        typer.echo(f"{spec.key}/canonical: existing artifacts verified")
        return

    repo.create_processing_run(
        canonical_run_id,
        "source-canonical-extraction",
        "1",
        git_sha(),
        canonical_digest,
        canonical_config,
    )
    shards, manifest = process_canonical_source(
        spec,
        raw_inputs,
        canonical_output_dir,
        target_words=target_words,
    )
    output_ids: list[str] = []
    total_words = total_bytes = total_records = total_newlines = 0
    for shard in shards:
        local = canonical_output_dir / shard.path
        uri = store.uri(f"processed/{spec.key}/canonical/{shard.path}")
        store.put_file(local, uri)
        if store.sha256(uri) != shard.physical_sha256:
            raise OSError(f"remote checksum mismatch: {uri}")
        artifact_id = "canonical-" + stable_id(
            spec.canonical_resource_version_id,
            shard.index,
            shard.physical_sha256,
        )
        with conn.transaction():
            repo.register_artifact(
                artifact_id,
                "canonical",
                uri,
                shard.physical_sha256,
                shard.compressed_bytes,
                "txt.zst",
                shard.record_count,
                spec.canonical_resource_version_id,
                integrity_status="verified",
                verification_report=asdict(shard),
            )
            repo.register_corpus_shard(
                spec.canonical_resource_version_id,
                shard.index,
                artifact_id,
                shard.word_count,
                shard.record_count,
                shard.compressed_bytes,
            )
        output_ids.append(artifact_id)
        total_words += shard.word_count
        total_bytes += shard.compressed_bytes
        total_records += shard.record_count
        total_newlines += shard.newline_count
    manifest_uri = store.uri(f"manifests/{spec.key}/canonical/manifest.json")
    store.put_file(manifest, manifest_uri)
    with conn.transaction():
        repo.upsert_corpus_version_stats(
            spec.canonical_resource_version_id,
            total_words,
            total_words + total_newlines,
            total_records,
            total_newlines,
            total_bytes,
            len(shards),
            {
                spec.key: total_words,
                "metrics": {
                    "canonical_words": total_words,
                    "newlines": total_newlines,
                    "word2vec_train_words": total_words + total_newlines,
                    "boundary_policy": boundary_policy,
                },
            },
            {},
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT artifact_id FROM artifacts WHERE resource_version_id=%s AND stage='raw' ORDER BY artifact_id",
                (spec.raw_resource_version_id,),
            )
            raw_artifact_ids = [row[0] for row in cur.fetchall()]
        repo.record_processing_io(canonical_run_id, raw_artifact_ids, output_ids)
        repo.finish_processing_run(
            canonical_run_id,
            "completed",
            diagnostics={
                "manifest_uri": manifest_uri,
                "logical_words": total_words,
                "newlines": total_newlines,
                "word2vec_train_words": total_words + total_newlines,
            },
        )
    for local in canonical_output_dir.iterdir():
        local.unlink()
    typer.echo(
        f"{spec.key}/canonical: COMPLETED ({total_words} words, {len(shards)} shards)"
    )


__all__ = [
    "_process_canonical_stage",
    "_raw_inputs",
]
