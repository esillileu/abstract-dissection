"""Operational CLI for the independent Common Crawl producer."""

from __future__ import annotations

import json
from typing import Any

import typer

from .corpus.cli import app as corpus_app
from .db.migrations import run_migrations
from .db.session import get_connection

app = typer.Typer(name="f2-cc", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")
TABLES = (
    "pipeline_runs",
    "candidate_records",
    "processing_results",
    "audit_assignments",
)


def _fingerprint(conn: Any, schema: str, table: str) -> tuple[int, int | None]:
    return conn.execute(
        f"SELECT count(*),bit_xor(hashtextextended(row_to_json(t)::text,0)) FROM {schema}.{table} t"
    ).fetchone()


@app.command("migrate")
def migrate() -> None:
    with get_connection() as conn:
        applied = run_migrations(conn)
    typer.echo(json.dumps({"database": "f2_cc", "applied": applied}, sort_keys=True))


@app.command("status")
def status() -> None:
    with get_connection() as conn:
        run_migrations(conn)
        state = {table: _fingerprint(conn, "cc", table) for table in TABLES}
        releases = conn.execute("SELECT count(*) FROM cc.releases").fetchone()[0]
    typer.echo(
        json.dumps(
            {"database": "f2_cc", "tables": state, "releases": releases}, sort_keys=True
        )
    )
