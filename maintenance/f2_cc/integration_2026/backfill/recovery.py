"""Document range recovery, validation, and shard generation."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from f2_cc.corpus.pipeline import PipelineRunner
from f2_cc.corpus.storage import ProvenanceExporter
from f2_cc.db.repository import CorpusStateRepository
from f2_cc.object_store import s3_config_from_environment
from repro_io.checksum import sha256_file
from repro_io.commoncrawl.fetcher import RangeFetcher
from repro_io.s3 import S3ObjectStore

from .models import Evidence


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


__all__ = ["recover"]
