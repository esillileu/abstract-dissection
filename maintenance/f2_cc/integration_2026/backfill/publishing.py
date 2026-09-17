"""Evidence upload to S3 and atomic lineage database application."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from f2_cc.object_store import s3_config_from_environment
from repro_io.s3 import S3ObjectStore

from .models import STAGES, Evidence, config_hash, stable_id


def upload_and_verify(evidence: list[Evidence]) -> None:
    store = S3ObjectStore(s3_config_from_environment())
    for item in evidence:
        store.put_file(Path(item.local_path), item.s3_uri)
        if (
            store.head(item.s3_uri).byte_size != item.byte_size
            or store.sha256(item.s3_uri) != item.sha256
        ):
            raise RuntimeError(f"uploaded artifact verification failed: {item.s3_uri}")


def apply_backfill(
    conn: Any, inspected: dict[str, Any], evidence: list[Evidence]
) -> None:
    run, run_id = inspected["run"], inspected["run"]["run_id"]
    cfg_hash = config_hash(run)
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
                        stable_id("stage", run_id, stage, cfg_hash, digest),
                        run_id,
                        stage,
                        cfg_hash,
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


__all__ = ["apply_backfill", "upload_and_verify"]
