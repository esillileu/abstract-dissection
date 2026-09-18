"""PostgreSQL connection management and transaction lifecycle for F2 Reproduction Catalog DB."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import psycopg

from f2.corpus.db.contract import (
    database_name,
    resolve_url,
    validate_connection,
    validate_database_url,
)


class CatalogDatabaseConfigError(Exception):
    """Raised when F2_DATABASE_URL is missing or invalid."""


@dataclass(frozen=True)
class CatalogDatabaseConfig:
    connection_url: str

    @classmethod
    def from_environment(cls) -> CatalogDatabaseConfig:
        return cls(
            connection_url=resolve_url(
                None,
                error_type=CatalogDatabaseConfigError,
            )
        )


def get_catalog_db_url() -> str:
    return CatalogDatabaseConfig.from_environment().connection_url


@contextmanager
def get_connection(
    connection_url: str | None = None,
    *,
    validate_contract: bool = False,
) -> Generator[psycopg.Connection[Any], None, None]:
    url = connection_url or get_catalog_db_url()
    validate_database_url(url, test=database_name(url) != "f2")
    with psycopg.connect(url, options="-c search_path=catalog,public") as conn:
        if validate_contract:
            validate_connection(
                conn,
                schema="catalog",
                latest_migration="001_initial_catalog_schema",
            )
        yield conn


@contextmanager
def transaction(
    connection_url: str | None = None,
    *,
    validate_contract: bool = False,
) -> Generator[psycopg.Connection[Any], None, None]:
    url = connection_url or get_catalog_db_url()
    validate_database_url(url, test=database_name(url) != "f2")
    with psycopg.connect(url, options="-c search_path=catalog,public") as conn:
        if validate_contract:
            validate_connection(
                conn,
                schema="catalog",
                latest_migration="001_initial_catalog_schema",
            )
        with conn.transaction():
            yield conn
