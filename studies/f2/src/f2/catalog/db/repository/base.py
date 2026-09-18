"""Base repository class providing connection management for F2 Catalog DB."""

from __future__ import annotations

from typing import Any

import psycopg


class BaseCatalogRepository:
    """Base repository holding database connection for catalog operations."""

    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        self.conn = conn
