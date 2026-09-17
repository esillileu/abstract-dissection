"""Operational CLI for the independent Common Crawl producer."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any

import psycopg
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
    "analysis_profiles",
)


def _f2_url() -> str:
    value = os.getenv("F2_DATABASE_URL")
    if not value:
        from dotenv import load_dotenv

        load_dotenv(override=True)
        value = os.getenv("F2_DATABASE_URL")
    if not value:
        raise RuntimeError("F2_DATABASE_URL is required for migration")
    return value


@contextmanager
def _source_connection():
    with psycopg.connect(_f2_url(), options="-c search_path=corpus,public") as conn:
        if conn.execute("SELECT current_database()").fetchone()[0] != "f2":
            raise RuntimeError("F2_DATABASE_URL must target database 'f2'")
        yield conn


def _fingerprint(conn: Any, schema: str, table: str) -> tuple[int, int | None]:
    return conn.execute(
        f"SELECT count(*),bit_xor(hashtextextended(row_to_json(t)::text,0)) FROM {schema}.{table} t"
    ).fetchone()


@app.command("migrate")
def migrate() -> None:
    with get_connection() as conn:
        applied = run_migrations(conn)
    typer.echo(json.dumps({"database": "f2_cc", "applied": applied}, sort_keys=True))


@app.command("import-f2")
def import_f2(apply: bool = typer.Option(False, "--apply")) -> None:
    """Copy legacy CC state from F2 and verify every table fingerprint."""
    with _source_connection() as source, get_connection() as target:
        run_migrations(target)
        before = {table: _fingerprint(source, "corpus", table) for table in TABLES}
        existing = {table: _fingerprint(target, "cc", table)[0] for table in TABLES}
        if any(existing.values()):
            raise RuntimeError(f"target CC tables must be empty: {existing}")
        if apply:
            with target.transaction():
                for table in TABLES:
                    with (
                        source.cursor().copy(
                            f"COPY corpus.{table} TO STDOUT"
                        ) as output,
                        target.cursor().copy(f"COPY cc.{table} FROM STDIN") as input_,
                    ):
                        for block in output:
                            input_.write(block)
                after = {table: _fingerprint(target, "cc", table) for table in TABLES}
                if after != before:
                    raise RuntimeError(
                        f"CC migration verification failed: source={before}, target={after}"
                    )
        typer.echo(
            json.dumps(
                {
                    "apply": apply,
                    "source_fingerprints": before,
                    "target_was_empty": existing,
                },
                sort_keys=True,
            )
        )


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
