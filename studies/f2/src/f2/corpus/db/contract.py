"""Shared F2 PostgreSQL connection contract.

This module contains only connection validation.  It deliberately does not
log or include connection URLs so authentication material cannot leak through
configuration errors.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import unquote, urlsplit

import psycopg


class DatabaseContractError(RuntimeError):
    """Raised when a connection is not the canonical F2 database contract."""


def database_name(connection_url: str) -> str:
    """Return the decoded database name without exposing the connection URL."""
    parsed = urlsplit(connection_url)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
        raise DatabaseContractError("F2 database URL must be a PostgreSQL URL")
    name = unquote(parsed.path.lstrip("/"))
    if not name or "/" in name:
        raise DatabaseContractError("F2 database URL must name one database")
    return name


def validate_database_url(connection_url: str, *, test: bool = False) -> str:
    """Reject aliases and unsafe test targets before opening a connection."""
    name = database_name(connection_url)
    if test:
        if name == "f2" or not name.lower().endswith(("test", "_test", "-test")):
            raise DatabaseContractError(
                "F2 test database URL must identify a dedicated test database"
            )
    elif name != "f2":
        raise DatabaseContractError(
            "F2 database URL must identify the canonical 'f2' database"
        )
    return connection_url


def resolve_url(
    explicit: str | None,
    *,
    error_type: type[Exception] = DatabaseContractError,
) -> str:
    """Resolve an F2 URL from an explicit value or the unified environment key."""
    if explicit and explicit.strip():
        return explicit.strip()

    url = os.getenv("F2_DATABASE_URL")
    if not url:
        try:
            from dotenv import load_dotenv

            load_dotenv(override=True)
            url = os.getenv("F2_DATABASE_URL")
        except Exception:
            # Configuration errors below remain deterministic when dotenv is
            # unavailable; dotenv is an optional convenience, not a source of
            # required credentials.
            pass
    if not url:
        raise error_type("F2_DATABASE_URL is required")
    value = url.strip()
    try:
        return validate_database_url(value)
    except DatabaseContractError as exc:
        raise error_type(str(exc)) from None


def _query_one(conn: psycopg.Connection[Any], query: str, params: tuple[Any, ...] = ()):
    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchone()


def validate_connection(
    conn: psycopg.Connection[Any],
    *,
    schema: str,
    required_schemas: tuple[str, ...] = ("catalog", "corpus"),
    latest_migration: str | None = None,
) -> None:
    """Validate database identity, owned schemas, and optionally migrations."""
    database = _query_one(conn, "SELECT current_database();")[0]
    if database != "f2":
        raise DatabaseContractError(
            "F2 connection rejected: current database must be 'f2'"
        )

    for required in required_schemas:
        present = _query_one(
            conn,
            "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = %s);",
            (required,),
        )[0]
        if not present:
            raise DatabaseContractError(
                f"F2 connection rejected: required schema '{required}' is missing"
            )

    if latest_migration is not None:
        present = _query_one(
            conn,
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name = 'schema_migrations');",
            (schema,),
        )[0]
        if not present:
            raise DatabaseContractError(
                f"F2 connection rejected: {schema} migrations are missing"
            )
        applied = _query_one(
            conn,
            f"SELECT EXISTS (SELECT 1 FROM {schema}.schema_migrations WHERE version = %s);",
            (latest_migration,),
        )[0]
        if not applied:
            raise DatabaseContractError(
                f"F2 connection rejected: {schema} migrations are incomplete"
            )


__all__ = [
    "DatabaseContractError",
    "database_name",
    "resolve_url",
    "validate_connection",
    "validate_database_url",
]
