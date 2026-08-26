"""PostgreSQL connection management and transaction lifecycle for F2 corpus pipeline."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import psycopg


class DatabaseConfigError(Exception):
    """Raised when F2_CORPUS_DATABASE_URL is missing or invalid."""


@dataclass(frozen=True)
class DatabaseConfig:
    connection_url: str

    @classmethod
    def from_environment(cls) -> DatabaseConfig:
        url = os.getenv("F2_CORPUS_DATABASE_URL")
        if not url:
            raise DatabaseConfigError(
                "Environment variable 'F2_CORPUS_DATABASE_URL' is required but not set."
            )
        return cls(connection_url=url)


def get_db_url() -> str:
    return DatabaseConfig.from_environment().connection_url


@contextmanager
def get_connection(
    connection_url: str | None = None,
) -> Generator[psycopg.Connection[Any], None, None]:
    url = connection_url or get_db_url()
    with psycopg.connect(url) as conn:
        yield conn


@contextmanager
def transaction(
    connection_url: str | None = None,
) -> Generator[psycopg.Connection[Any], None, None]:
    url = connection_url or get_db_url()
    with psycopg.connect(url) as conn:
        with conn.transaction():
            yield conn
