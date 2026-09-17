"""Stage 2 processing orchestrator: canonical shards to word2vec normalized shards."""

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
    process_normalized_source,
    tool_versions,
)
from ..sources import CorpusSource, stable_id


def _canonical_inputs(
    repo: CorpusStateRepository, spec: CorpusSource, store: S3ObjectStore, target: Path
) -> list[Path]:
    target.mkdir(parents=True, exist_ok=True)
    with repo.conn.cursor() as cur:
        cur.execute(
            "SELECT s3_uri, sha256 FROM artifacts WHERE resource_version_id=%s AND stage='canonical' ORDER BY s3_uri",
            (spec.canonical_resource_version_id,),
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
            raise OSError(f"downloaded canonical shard checksum mismatch: {uri}")
        result.append(local)
    return result


def _process_normalized_stage(
    conn: psycopg.Connection,
    repo: CorpusStateRepository,
    spec: CorpusSource,
    store: S3ObjectStore,
    paths: RuntimePaths,
    target_words: int,
    boundary_policy: str,
) -> None:
    canonical_cache_dir = (
        paths.staging_root / "exp" / "f2" / "sources" / spec.key / "canonical-cache"
    )
    canonical_inputs = _canonical_inputs(repo, spec, store, canonical_cache_dir)
    if not canonical_inputs:
        raise RuntimeError("no verified canonical artifacts found")

    normalized_config = {
        "source": spec.key,
        "stage": "word2vec_public_normalized_v1",
        "recipe": "mikolov_demo_train_big_model_v1_normalize_text",
        "target_words": target_words,
        "boundary_policy": boundary_policy,
        "tools": tool_versions(),
    }
    normalized_digest = config_hash(normalized_config)
    normalized_run_id = "proc-" + stable_id(
        spec.normalized_resource_version_id, normalized_digest
    )
    previous_normalized = repo.get_processing_run(normalized_run_id)
    normalized_stats = repo.get_corpus_version_stats(
        spec.normalized_resource_version_id
    )
    normalized_output_dir = (
        paths.staging_root / "exp" / "f2" / "sources" / spec.key / "normalized"
    )

    if (
        previous_normalized
        and previous_normalized["status"] == "completed"
        and normalized_stats
    ):
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM processing_run_inputs WHERE run_id=%s",
                (normalized_run_id,),
            )
            in_count = cur.fetchone()[0]
        if in_count == 0:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT artifact_id FROM artifacts WHERE resource_version_id=%s AND stage='canonical' ORDER BY artifact_id",
                    (spec.canonical_resource_version_id,),
                )
                c_ids = [row[0] for row in cur.fetchall()]
                cur.execute(
                    "SELECT artifact_id FROM artifacts WHERE resource_version_id=%s AND stage='normalized' ORDER BY artifact_id",
                    (spec.normalized_resource_version_id,),
                )
                n_ids = [row[0] for row in cur.fetchall()]
            repo.record_processing_io(normalized_run_id, c_ids, n_ids)
        typer.echo(f"{spec.key}/normalized: already completed")
        return

    repo.create_processing_run(
        normalized_run_id,
        "word2vec-public-normalization-v1",
        "1",
        git_sha(),
        normalized_digest,
        normalized_config,
    )
    shards, manifest = process_normalized_source(
        spec,
        canonical_inputs,
        normalized_output_dir,
        target_words=target_words,
    )
    output_ids: list[str] = []
    total_words = total_bytes = total_records = total_newlines = 0
    for shard in shards:
        local = normalized_output_dir / shard.path
        uri = store.uri(f"processed/{spec.key}/normalized/{shard.path}")
        store.put_file(local, uri)
        if store.sha256(uri) != shard.physical_sha256:
            raise OSError(f"remote checksum mismatch: {uri}")
        artifact_id = "normalized-" + stable_id(
            spec.normalized_resource_version_id,
            shard.index,
            shard.physical_sha256,
        )
        with conn.transaction():
            repo.register_artifact(
                artifact_id,
                "normalized",
                uri,
                shard.physical_sha256,
                shard.compressed_bytes,
                "txt.zst",
                shard.record_count,
                spec.normalized_resource_version_id,
                integrity_status="verified",
                verification_report=asdict(shard),
            )
            repo.register_corpus_shard(
                spec.normalized_resource_version_id,
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
    manifest_uri = store.uri(f"manifests/{spec.key}/normalized/manifest.json")
    store.put_file(manifest, manifest_uri)
    with conn.transaction():
        repo.upsert_corpus_version_stats(
            spec.normalized_resource_version_id,
            total_words,
            total_words + total_newlines,
            total_records,
            total_newlines,
            total_bytes,
            len(shards),
            {
                spec.key: total_words,
                "metrics": {
                    "normalized_lexical_words": total_words,
                    "newlines": total_newlines,
                    "word2vec_train_words": total_words + total_newlines,
                    "boundary_policy": boundary_policy,
                },
            },
            {},
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT artifact_id FROM artifacts WHERE resource_version_id=%s AND stage='canonical' ORDER BY artifact_id",
                (spec.canonical_resource_version_id,),
            )
            canonical_artifact_ids = [row[0] for row in cur.fetchall()]
        repo.record_processing_io(normalized_run_id, canonical_artifact_ids, output_ids)
        repo.finish_processing_run(
            normalized_run_id,
            "completed",
            diagnostics={
                "manifest_uri": manifest_uri,
                "logical_words": total_words,
                "newlines": total_newlines,
                "word2vec_train_words": total_words + total_newlines,
            },
        )
    for local in normalized_output_dir.iterdir():
        local.unlink()
    typer.echo(
        f"{spec.key}/normalized: COMPLETED ({total_words} words, {len(shards)} shards)"
    )


__all__ = [
    "_canonical_inputs",
    "_process_normalized_stage",
]
