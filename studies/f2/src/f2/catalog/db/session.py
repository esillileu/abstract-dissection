"""PostgreSQL connection management and transaction lifecycle for F2 Reproduction Catalog DB."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import psycopg


class CatalogDatabaseConfigError(Exception):
    """Raised when F2_CATALOG_DATABASE_URL is missing or invalid."""


@dataclass(frozen=True)
class CatalogDatabaseConfig:
    connection_url: str

    @classmethod
    def from_environment(cls) -> CatalogDatabaseConfig:
        url = os.getenv("F2_CATALOG_DATABASE_URL")
        if not url:
            try:
                from dotenv import load_dotenv

                load_dotenv()
                url = os.getenv("F2_CATALOG_DATABASE_URL")
            except Exception:
                pass
        if not url:
            url = "postgresql://f2_catalog:f2_catalog@localhost:5432/f2_catalog"
        return cls(connection_url=url)


def get_catalog_db_url() -> str:
    return CatalogDatabaseConfig.from_environment().connection_url


@contextmanager
def get_connection(
    connection_url: str | None = None,
) -> Generator[psycopg.Connection[Any], None, None]:
    url = connection_url or get_catalog_db_url()
    with psycopg.connect(url) as conn:
        yield conn


@contextmanager
def transaction(
    connection_url: str | None = None,
) -> Generator[psycopg.Connection[Any], None, None]:
    url = connection_url or get_catalog_db_url()
    with psycopg.connect(url) as conn:
        with conn.transaction():
            yield conn
