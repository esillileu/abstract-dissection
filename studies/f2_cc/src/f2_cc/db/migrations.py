"""Immutable F2-CC schema migration runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_migrations(conn: Any) -> list[str]:
    applied: list[str] = []
    with conn.transaction():
        conn.execute("CREATE SCHEMA IF NOT EXISTS cc")
        conn.execute("SET LOCAL search_path=cc,public")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations(version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"
        )
        existing = {
            row[0] for row in conn.execute("SELECT version FROM schema_migrations")
        }
        for path in sorted((Path(__file__).parent / "sql").glob("*.sql")):
            if path.stem not in existing:
                conn.execute(path.read_text(encoding="utf-8"))
                conn.execute(
                    "INSERT INTO schema_migrations(version) VALUES(%s)", (path.stem,)
                )
                applied.append(path.stem)
    return applied
