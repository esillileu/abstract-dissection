"""Evidence data models and hashing helpers for Common Crawl backfill."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

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


def config_hash(run: dict[str, Any]) -> str:
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


__all__ = ["STAGES", "Evidence", "config_hash", "stable_id"]
