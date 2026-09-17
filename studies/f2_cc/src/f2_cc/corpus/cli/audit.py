from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from repro_core.context.paths import RuntimePaths

from ...db.repository import CorpusStateRepository
from ...db.session import get_connection
from ..discovery import SequentialAuditSampler


def create_audit(
    run_id: Annotated[
        str,
        typer.Option("--run-id", "-r", help="Run ID from which to create audit sample"),
    ],
    budget: Annotated[
        int, typer.Option("--budget", "-b", help="Total audit sample size")
    ] = 400,
    seed: Annotated[
        int, typer.Option("--seed", "-s", help="Random seed for audit priority")
    ] = 20260227,
) -> None:
    """Generate pre-specified 8-stratum sequential audit assignments and store in PostgreSQL."""
    with get_connection() as conn:
        repo = CorpusStateRepository(conn)
        records = repo.get_provenance_records(run_id)
        if not records:
            typer.echo(f"No records found for run '{run_id}'.")
            return

        # Use only records that were actually fetched
        fetched_records = [r for r in records if r.get("fetch_status") == "success"]
        sampler = SequentialAuditSampler(seed=seed)
        schedule = sampler.generate_8_stratum_audit_schedule(fetched_records)
        assignments = sampler.select_8_stratum_audit_wave(schedule, total_budget=budget)

        count = repo.insert_audit_assignments(run_id, assignments)
        typer.echo(
            f"Created {count} audit assignments across 8 design strata for run '{run_id}'."
        )


def record_audit(
    run_id: Annotated[
        str, typer.Option("--run-id", "-r", help="Run ID to record audit labels for")
    ],
    audit_file: Annotated[
        Path | None,
        typer.Option("--audit-file", "-a", help="Path to annotated audit JSONL"),
    ] = None,
) -> None:
    """Record completed audit gold labels into PostgreSQL from an annotated JSONL file."""
    paths = RuntimePaths.from_environment()
    target_audit_file = audit_file
    if target_audit_file is None:
        candidate_audits = [
            paths.staging_root
            / "exp"
            / "f2"
            / "00_corpus_audit_set_50k_400_annotated.jsonl",
            paths.staging_root / "exp" / "f2" / "audit_set_50k_400_annotated.jsonl",
            paths.analysis_output("f2") / "00_corpus_audit_set_50k_400_annotated.jsonl",
            paths.analysis_output("f2") / "audit_set_50k_400_annotated.jsonl",
        ]
        target_audit_file = next(
            (p for p in candidate_audits if p.exists()), candidate_audits[0]
        )

    if not target_audit_file.exists():
        typer.echo(f"Audit file not found: {target_audit_file}")
        return

    annotated_records = [
        json.loads(line)
        for line in target_audit_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with get_connection() as conn:
        repo = CorpusStateRepository(conn)
        recorded = 0
        for r in annotated_records:
            cand_id = r.get("candidate_id") or r.get("record_id")
            if r.get("gold_class") is not None and r.get("word_count_gold") is not None:
                repo.record_audit_gold_label(
                    run_id=run_id,
                    candidate_id=cand_id,
                    gold_class=int(r["gold_class"]),
                    word_count_gold=int(r["word_count_gold"]),
                    auditor_id=r.get("auditor_id", "blind_expert"),
                    notes=r.get("notes"),
                )
                recorded += 1
        typer.echo(
            f"Recorded {recorded} gold audit labels into PostgreSQL for run '{run_id}'."
        )
