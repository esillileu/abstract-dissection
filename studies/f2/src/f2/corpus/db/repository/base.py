"""Base repository class providing database connection and commit semantics."""

from __future__ import annotations

from typing import Any

import psycopg


class BaseCorpusRepository:
    """Base state repository with connection lifecycle management."""

    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        self.conn = conn

    def _commit(self) -> None:
        """Commit unless an outer transaction owns the connection."""
        if not getattr(self.conn, "_num_transactions", 0):
            self.conn.commit()


__all__ = ["BaseCorpusRepository"]
