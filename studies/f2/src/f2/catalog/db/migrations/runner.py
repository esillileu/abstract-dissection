"""Migration runner to apply versioned SQL migrations idempotently for F2 Catalog DB."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg


def get_migrations_dir() -> Path:
    return Path(__file__).parent


def run_catalog_migrations(conn: psycopg.Connection[Any]) -> list[str]:
    """Execute all pending SQL migrations in ascending order inside a transaction."""
    applied: list[str] = []
    migrations_dir = get_migrations_dir()

    with conn.transaction():
        with conn.cursor() as cur:
            # Ensure schema_migrations table exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version VARCHAR(64) PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            cur.execute("SELECT version FROM schema_migrations;")
            already_applied = {row[0] for row in cur.fetchall()}

            # Find all .sql files in sorted order
            sql_files = sorted(migrations_dir.glob("*.sql"))

            for sql_file in sql_files:
                version = sql_file.stem
                if version not in already_applied:
                    sql_content = sql_file.read_text(encoding="utf-8")
                    cur.execute(sql_content)
                    cur.execute(
                        "INSERT INTO schema_migrations (version) VALUES (%s);",
                        (version,),
                    )
                    applied.append(version)

    return applied
