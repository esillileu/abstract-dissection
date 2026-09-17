"""PostgreSQL connection management and transaction lifecycle for F2 corpus pipeline."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import psycopg

from .contract import (
    database_name,
    resolve_url,
    validate_connection,
    validate_database_url,
)


class DatabaseConfigError(Exception):
    """Raised when F2_DATABASE_URL is missing or invalid."""


@dataclass(frozen=True)
class DatabaseConfig:
    connection_url: str

    @classmethod
    def from_environment(cls) -> DatabaseConfig:
        return cls(
            connection_url=resolve_url(
                None,
                error_type=DatabaseConfigError,
            )
        )


def get_db_url() -> str:
    return DatabaseConfig.from_environment().connection_url


@contextmanager
def get_connection(
    connection_url: str | None = None,
    *,
    validate_contract: bool = False,
) -> Generator[psycopg.Connection[Any], None, None]:
    url = connection_url or get_db_url()
    validate_database_url(url, test=database_name(url) != "f2")
    with psycopg.connect(url, options="-c search_path=corpus,public") as conn:
        if validate_contract:
            validate_connection(
                conn,
                schema="corpus",
                latest_migration="006_remove_common_crawl_operational_state",
            )
        yield conn


@contextmanager
def transaction(
    connection_url: str | None = None,
    *,
    validate_contract: bool = False,
) -> Generator[psycopg.Connection[Any], None, None]:
    url = connection_url or get_db_url()
    validate_database_url(url, test=database_name(url) != "f2")
    with psycopg.connect(url, options="-c search_path=corpus,public") as conn:
        if validate_contract:
            validate_connection(
                conn,
                schema="corpus",
                latest_migration="006_remove_common_crawl_operational_state",
            )
        with conn.transaction():
            yield conn
