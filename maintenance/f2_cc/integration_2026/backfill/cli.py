"""Command-line interface for Common Crawl integration backfill."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from f2_cc.db.session import get_connection

from repro_core.context.paths import RuntimePaths

from ..contracts import CANONICAL_PROFILES
from .inspection import inspect_run
from .models import Evidence
from .publishing import apply_backfill, upload_and_verify
from .recovery import recover


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


__all__ = ["main"]
