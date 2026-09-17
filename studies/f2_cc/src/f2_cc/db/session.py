"""Connection contract for the physically separate F2-CC database."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import psycopg


class F2CCDatabaseConfigError(RuntimeError):
    pass


def get_db_url() -> str:
    url = os.getenv("F2_CC_DATABASE_URL")
    if not url:
        try:
            from dotenv import load_dotenv

            load_dotenv(override=True)
            url = os.getenv("F2_CC_DATABASE_URL")
        except Exception:
            pass
    if not url:
        raise F2CCDatabaseConfigError("F2_CC_DATABASE_URL is required")
    return url.strip()


@contextmanager
def get_connection(
    connection_url: str | None = None,
) -> Generator[psycopg.Connection[Any], None, None]:
    with psycopg.connect(
        connection_url or get_db_url(), options="-c search_path=cc,public"
    ) as conn:
        database = conn.execute("SELECT current_database()").fetchone()[0]
        if database != "f2_cc" and not database.lower().endswith(
            ("test", "_test", "-test")
        ):
            raise F2CCDatabaseConfigError(
                "F2_CC_DATABASE_URL must target database 'f2_cc'"
            )
        yield conn
