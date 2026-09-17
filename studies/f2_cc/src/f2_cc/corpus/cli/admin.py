from __future__ import annotations

import typer

from ...db.migrations import run_migrations
from ...db.session import get_connection


def migrate_db() -> None:
    """Apply pending PostgreSQL migrations for the F2 corpus control plane."""
    with get_connection() as conn:
        applied = run_migrations(conn)
        if applied:
            typer.echo(
                f"Successfully applied {len(applied)} migrations: {', '.join(applied)}"
            )
        else:
            typer.echo("Database schema is up to date (0 pending migrations).")
