"""Read-only F2-CC migration and release evidence checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from f2_cc.db.session import get_connection
from f2_cc.object_store import s3_config_from_environment
from repro_io.s3 import S3ObjectStore

from .cleanup import candidate_rows
from .contracts import CANONICAL_PROFILES, EXPECTED_CLEANUP_RUNS


def database_checks(conn: Any) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, observed: Any, expected: Any) -> None:
        checks.append(
            {
                "name": name,
                "status": "PASS" if observed == expected else "FAIL",
                "observed": observed,
                "expected": expected,
            }
        )

    with conn.cursor() as cur:
        cur.execute(
            "SELECT profile_key,run_id FROM analysis_profiles WHERE is_active ORDER BY profile_key"
        )
        add("canonical_profiles", dict(cur.fetchall()), CANONICAL_PROFILES)
        for run_id in CANONICAL_PROFILES.values():
            cur.execute(
                "SELECT sample_size FROM pipeline_runs WHERE run_id=%s", (run_id,)
            )
            sample_size = cur.fetchone()[0]
            cur.execute(
                "SELECT count(*) FROM candidate_records WHERE run_id=%s", (run_id,)
            )
            add(f"{run_id}_candidates", cur.fetchone()[0], sample_size)
            cur.execute(
                "SELECT count(*) FROM processing_results WHERE run_id=%s", (run_id,)
            )
            add(f"{run_id}_processing", cur.fetchone()[0], sample_size)
    add("cleanup_candidate_count", len(candidate_rows(conn)), EXPECTED_CLEANUP_RUNS)
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--s3", action="store_true")
    args = parser.parse_args()
    with get_connection() as conn:
        checks = database_checks(conn)
    report: dict[str, Any] = {"checks": checks, "credentials_recorded": False}
    if args.s3:
        store = S3ObjectStore(s3_config_from_environment())
        report["s3_objects"] = [
            item.__dict__ for item in store.list(store.config.root_uri)
        ]
    report["status"] = (
        "FAIL" if any(item["status"] == "FAIL" for item in checks) else "PASS"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
