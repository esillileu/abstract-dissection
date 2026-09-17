"""Base repository class providing connection and transaction management."""

from __future__ import annotations

from typing import Any

import psycopg


class BaseRepository:
    """Base repository holding database connection and transaction helpers."""

    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        self.conn = conn

    def _commit(self) -> None:
        """Commit the current transaction if not managed by an outer transaction block."""
        if not getattr(self.conn, "_num_transactions", 0):
            self.conn.commit()
