"""Recover canonical Common Crawl outputs and atomically backfill their lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from f2_cc.corpus.pipeline import PipelineRunner
from f2_cc.corpus.storage import ProvenanceExporter
from f2_cc.db.repository import CorpusStateRepository
from f2_cc.db.session import get_connection
from f2_cc.object_store import s3_config_from_environment
from repro_io.checksum import sha256_file
from repro_io.commoncrawl.fetcher import RangeFetcher
from repro_io.s3 import S3ObjectStore

from repro_core.context.paths import RuntimePaths

from .contracts import CANONICAL_PROFILES

STAGES = (
    ("cdx_discovery", "acquisition"),
    ("arc_range_fetch", "acquisition"),
    ("html_extract", "processing"),
    ("reject_explore", "processing"),
    ("shard_text", "processing"),
)


@dataclass(frozen=True)
class Evidence:
    role: str
    local_path: str
    s3_uri: str
    sha256: str
    byte_size: int
    record_count: int
    format: str


def stable_id(kind: str, run_id: str, stage: str, config_hash: str, digest: str) -> str:
    value = f"{run_id}:{stage}:{config_hash}:{digest}"
    return f"{kind}-" + hashlib.sha256(value.encode()).hexdigest()[:32]


def inspect_run(conn: Any, profile: str) -> dict[str, Any]:
    run_id = CANONICAL_PROFILES[profile]
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM pipeline_runs WHERE run_id=%s", (run_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"missing canonical run {run_id}")
        run = dict(zip((item.name for item in cur.description), row, strict=True))
        cur.execute(
            """SELECT c.candidate_id,c.crawl_id,c.url,c.arc_filename,c.arc_offset,
                       c.arc_length,c.inclusion_probability,c.design_weight,c.block_index,
                       c.record_index_in_block,r.clean_text_sha256,r.word_count
                       FROM candidate_records c JOIN processing_results r USING(run_id,candidate_id)
                       WHERE c.run_id=%s AND r.clean_text_sha256 IS NOT NULL
                       ORDER BY array_position(%s::text[],c.crawl_id),c.block_index,
                                c.record_index_in_block,c.candidate_id""",
            (run_id, run["crawl_ids"]),
        )
        columns = [item.name for item in cur.description]
        documents = [dict(zip(columns, item, strict=True)) for item in cur.fetchall()]
        cur.execute(
            "SELECT stage_role FROM stage_lineage WHERE source_run_id=%s", (run_id,)
        )
        lineage = [item[0] for item in cur.fetchall()]
    return {"profile": profile, "run": run, "documents": documents, "lineage": lineage}


def _recover_one(document: dict[str, Any], fetcher: RangeFetcher, cache: Path) -> Path:
    target = cache / f"{document['candidate_id']}.txt"
    if target.exists() and sha256_file(target) == document["clean_text_sha256"]:
        return target
    fetched = fetcher.fetch_range(
        document["arc_filename"], document["arc_offset"], document["arc_length"]
    )
    if fetched.status_code not in {200, 206} or not fetched.data:
        raise RuntimeError(
            f"range recovery failed for {document['candidate_id']}: {fetched.error_message}"
        )
    result = PipelineRunner().process(
        document["candidate_id"],
        document["crawl_id"],
        document["url"],
        fetched.data,
        document["inclusion_probability"],
        document["design_weight"],
        fetched.downloaded_bytes,
    )
    if (
        result.clean_text is None
        or hashlib.sha256(result.clean_text.encode()).hexdigest()
        != document["clean_text_sha256"]
    ):
        raise RuntimeError(f"clean-text digest mismatch for {document['candidate_id']}")
    if result.word_count != document["word_count"]:
        raise RuntimeError(f"word-count mismatch for {document['candidate_id']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result.clean_text, encoding="utf-8")
    return target


def recover(conn: Any, inspected: dict[str, Any], root: Path) -> list[Evidence]:
    run, documents = inspected["run"], inspected["documents"]
    output, cache = root / run["run_id"], root / run["run_id"] / "documents"
    fetcher = RangeFetcher(
        user_agent="abstract-dissection-repro/0.1 (Research reproduction study)",
        bandwidth_mbps=run["bandwidth_mbps"],
        max_concurrency=max(1, run["concurrency"]),
    )
    with ThreadPoolExecutor(max_workers=max(1, run["concurrency"])) as pool:
        paths = list(
            pool.map(lambda item: _recover_one(item, fetcher, cache), documents)
        )
    shard = output / "clean_shards" / "shard_00000.txt"
    shard.parent.mkdir(parents=True, exist_ok=True)
    with shard.open("w", encoding="utf-8") as stream:
        for document, path in zip(documents, paths, strict=True):
            stream.write(
                f'<DOC url="{document["url"]}" words="{document["word_count"]}">\n'
            )
            stream.write(path.read_text(encoding="utf-8").strip())
            stream.write("\n</DOC>\n\n")
    exports = ProvenanceExporter(CorpusStateRepository(conn)).export(
        run["run_id"], output
    )
    store = S3ObjectStore(s3_config_from_environment())
    specs = (
        ("clean_shard", shard, len(documents), "txt"),
        ("provenance_jsonl", exports["jsonl"], run["sample_size"], "jsonl"),
        ("provenance_parquet", exports["parquet"], run["sample_size"], "parquet"),
    )
    evidence = [
        Evidence(
            role,
            str(path),
            store.uri(f"integration_2026/{run['run_id']}/{path.name}"),
            sha256_file(path),
            path.stat().st_size,
            count,
            format_name,
        )
        for role, path, count, format_name in specs
    ]
    release_manifest = output / "release.json"
    release_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": inspected["profile"],
                "source_run_id": run["run_id"],
                "artifacts": [asdict(item) for item in evidence],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    evidence.append(
        Evidence(
            "release_manifest",
            str(release_manifest),
            store.uri(f"integration_2026/{run['run_id']}/release.json"),
            sha256_file(release_manifest),
            release_manifest.stat().st_size,
            len(evidence),
            "json",
        )
    )
    return evidence


def upload_and_verify(evidence: list[Evidence]) -> None:
    store = S3ObjectStore(s3_config_from_environment())
    for item in evidence:
        store.put_file(Path(item.local_path), item.s3_uri)
        if (
            store.head(item.s3_uri).byte_size != item.byte_size
            or store.sha256(item.s3_uri) != item.sha256
        ):
            raise RuntimeError(f"uploaded artifact verification failed: {item.s3_uri}")


def _config_hash(run: dict[str, Any]) -> str:
    metadata = run.get("metadata") or {}
    return (
        metadata.get("config_hash")
        or hashlib.sha256(
            json.dumps(
                {
                    key: run[key]
                    for key in (
                        "crawl_ids",
                        "sample_size",
                        "seed",
                        "bandwidth_mbps",
                        "concurrency",
                    )
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
    )


def apply_backfill(
    conn: Any, inspected: dict[str, Any], evidence: list[Evidence]
) -> None:
    run, run_id = inspected["run"], inspected["run"]["run_id"]
    config_hash = _config_hash(run)
    digest = hashlib.sha256(
        json.dumps([asdict(item) for item in evidence], sort_keys=True).encode()
    ).hexdigest()
    release_id = f"cc-{inspected['profile']}-verified-v1"
    manifest = next(item for item in evidence if item.role == "release_manifest")
    with conn.transaction():
        with conn.cursor() as cur:
            for stage, _kind in STAGES:
                cur.execute(
                    """INSERT INTO stage_lineage(stage_id,source_run_id,stage_role,config_hash,evidence_digest,status,metadata)
                       VALUES(%s,%s,%s,%s,%s,'verified',%s::jsonb)
                       ON CONFLICT(stage_id) DO UPDATE SET status='verified',metadata=EXCLUDED.metadata""",
                    (
                        stable_id("stage", run_id, stage, config_hash, digest),
                        run_id,
                        stage,
                        config_hash,
                        digest,
                        json.dumps({"integration": "2026"}),
                    ),
                )
            cur.execute(
                """INSERT INTO releases(release_id,profile_key,source_run_id,manifest_uri,manifest_sha256,status,statistics,published_at)
                   VALUES(%s,%s,%s,%s,%s,'published',%s::jsonb,NOW())
                   ON CONFLICT(release_id) DO UPDATE SET manifest_uri=EXCLUDED.manifest_uri,
                     manifest_sha256=EXCLUDED.manifest_sha256,status='published',statistics=EXCLUDED.statistics,published_at=NOW()""",
                (
                    release_id,
                    inspected["profile"],
                    run_id,
                    manifest.s3_uri,
                    manifest.sha256,
                    json.dumps(
                        {
                            "sample_size": run["sample_size"],
                            "accepted_documents": len(inspected["documents"]),
                        }
                    ),
                ),
            )
            for item in evidence:
                cur.execute(
                    """INSERT INTO release_artifacts(release_id,role,uri,sha256,byte_size,record_count,format)
                       VALUES(%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT(release_id,role,uri) DO UPDATE SET sha256=EXCLUDED.sha256,
                         byte_size=EXCLUDED.byte_size,record_count=EXCLUDED.record_count,format=EXCLUDED.format""",
                    (
                        release_id,
                        item.role,
                        item.s3_uri,
                        item.sha256,
                        item.byte_size,
                        item.record_count,
                        item.format,
                    ),
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", action="append", choices=sorted(CANONICAL_PROFILES)
    )
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    profiles = args.profile or sorted(CANONICAL_PROFILES)
    root = (
        RuntimePaths.from_environment().staging_root
        / "maintenance/f2_cc/integration_2026"
    )
    plans = []
    with get_connection() as conn:
        for profile in profiles:
            inspected = inspect_run(conn, profile)
            evidence_path = root / inspected["run"]["run_id"] / "evidence.json"
            if args.recover:
                evidence = recover(conn, inspected, root)
                evidence_path.write_text(
                    json.dumps([asdict(x) for x in evidence], indent=2, sort_keys=True)
                    + "\n"
                )
            else:
                evidence = (
                    [Evidence(**x) for x in json.loads(evidence_path.read_text())]
                    if evidence_path.exists()
                    else []
                )
            if args.apply:
                if not evidence:
                    raise SystemExit(f"missing evidence for {profile}; run --recover")
                upload_and_verify(evidence)
                apply_backfill(conn, inspected, evidence)
            plans.append(
                {
                    "profile": profile,
                    "run_id": inspected["run"]["run_id"],
                    "accepted_documents": len(inspected["documents"]),
                    "existing_lineage": inspected["lineage"],
                    "evidence": [asdict(x) for x in evidence],
                }
            )
    print(json.dumps({"apply": args.apply, "plans": plans}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
