"""Create and apply the exact reviewable pipeline-run cleanup manifest."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from f2_cc.db.session import get_connection

from .contracts import EXPECTED_CLEANUP_RUNS, manifest_hash


def candidate_rows(conn: Any) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT p.run_id, p.run_type,
              CASE WHEN p.run_type IN ('test', 'smoke', 'test_audit') THEN 'explicit_test_type'
                   WHEN p.output_dir LIKE '/tmp/pytest-of-esillileu/%%' THEN 'pytest_output'
                   WHEN p.run_id LIKE 'smoke_%%_test' THEN 'smoke_test_id' END reason,
              COUNT(DISTINCT c.candidate_id) candidate_rows,
              COUNT(DISTINCT r.candidate_id) processing_rows,
              COUNT(DISTINCT a.audit_id) audit_rows
            FROM pipeline_runs p
            LEFT JOIN candidate_records c ON c.run_id=p.run_id
            LEFT JOIN processing_results r ON r.run_id=p.run_id
            LEFT JOIN audit_assignments a ON a.run_id=p.run_id
            WHERE (p.run_type IN ('test','smoke','test_audit')
                   OR p.output_dir LIKE '/tmp/pytest-of-esillileu/%%'
                   OR p.run_id LIKE 'smoke_%%_test')
              AND NOT EXISTS (SELECT 1 FROM analysis_profiles x WHERE x.run_id=p.run_id)
              AND NOT EXISTS (SELECT 1 FROM stage_lineage x WHERE x.source_run_id=p.run_id)
            GROUP BY p.run_id,p.run_type,reason ORDER BY p.run_id
        """)
        columns = [item.name for item in cur.description]
        return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]


def manifest_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "kind": "f2-pipeline-run-cleanup",
        "created_at": datetime.now(UTC).isoformat(),
        "expected_count": EXPECTED_CLEANUP_RUNS,
        "s3_deletion": False,
        "runs": rows,
    }


def validate_manifest(conn: Any, payload: dict[str, Any], digest: str) -> list[str]:
    if manifest_hash(payload) != digest:
        raise ValueError("cleanup manifest hash does not match")
    runs = payload.get("runs")
    if payload.get("kind") != "f2-pipeline-run-cleanup" or not isinstance(runs, list):
        raise ValueError("invalid cleanup manifest")
    if (
        payload.get("expected_count") != EXPECTED_CLEANUP_RUNS
        or len(runs) != EXPECTED_CLEANUP_RUNS
    ):
        raise ValueError(f"cleanup candidate count must equal {EXPECTED_CLEANUP_RUNS}")
    selected = [row.get("run_id") for row in runs]
    current = [row["run_id"] for row in candidate_rows(conn)]
    if selected != current or len(selected) != len(set(selected)):
        raise ValueError(
            "cleanup manifest does not exactly match current safe candidates"
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-hash")
    parser.add_argument("--backup-evidence")
    parser.add_argument("--restore-evidence")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    with get_connection() as conn:
        if args.manifest:
            payload = json.loads(args.manifest.read_text(encoding="utf-8"))
            embedded = payload.pop("manifest_sha256", None)
            run_ids = validate_manifest(
                conn, payload, args.manifest_hash or embedded or ""
            )
            if args.apply:
                if not args.backup_evidence or not args.restore_evidence:
                    raise SystemExit(
                        "apply requires backup and restore rehearsal evidence identifiers"
                    )
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute(
                            "DELETE FROM pipeline_runs WHERE run_id = ANY(%s)",
                            (run_ids,),
                        )
            print(
                json.dumps(
                    {"apply": args.apply, "runs": len(run_ids), "s3_deletion": False}
                )
            )
            return
        if not args.output:
            parser.error("--output is required when creating a manifest")
        payload = manifest_payload(candidate_rows(conn))
        digest = manifest_hash(payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload | {"manifest_sha256": digest}, indent=2, sort_keys=True)
            + "\n"
        )
        print(
            json.dumps(
                {
                    "manifest": str(args.output),
                    "sha256": digest,
                    "runs": len(payload["runs"]),
                }
            )
        )


if __name__ == "__main__":
    main()
