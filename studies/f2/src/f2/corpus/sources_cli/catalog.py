"""CLI commands for preflight checks, catalog source registration, and lifecycle status."""

from __future__ import annotations

import json

import typer

from repro_core.context.paths import RuntimePaths

from ..db.migrations.runner import run_migrations
from ..db.repository import CorpusStateRepository
from ..db.session import get_connection
from ..lifecycle import catalog_sources, install_validation_profiles, preflight
from ..sources import SOURCES, VALIDATION_PROFILES
from .common import _store


def sources_preflight() -> None:
    """Check scratch, tools, database, and SeaweedFS without exposing secrets."""
    paths = RuntimePaths.from_environment()
    result = preflight(paths.staging_root, _store())
    with get_connection() as conn:
        run_migrations(conn)
        conn.execute("SELECT 1")
    result["database"] = "reachable"
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


def sources_catalog() -> None:
    """Register all frozen source releases and immutable validation profiles."""
    from f2.catalog.db.migrations.runner import run_catalog_migrations

    with get_connection() as conn:
        run_catalog_migrations(conn)
        run_migrations(conn)
        catalog_sources(conn)
        install_validation_profiles(CorpusStateRepository(conn))
    typer.echo(
        f"cataloged {len(SOURCES)} sources and {len(VALIDATION_PROFILES)} validation profiles"
    )


def sources_status() -> None:
    """Print database-derived lifecycle status as JSON."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT r.resource_id, r.acquisition_status, r.readiness_status, rv.resource_version_id,
                       COALESCE(cvs.total_words,0), COALESCE(cvs.total_shards,0)
                       FROM catalog.resources r JOIN catalog.resource_versions rv USING(resource_id)
                       LEFT JOIN corpus.corpus_version_stats cvs USING(resource_version_id)
                       WHERE r.resource_id LIKE 'f2-%' ORDER BY r.resource_id, rv.resource_version_id""")
            rows = [
                {
                    "resource_id": row[0],
                    "acquisition": row[1],
                    "readiness": row[2],
                    "resource_version_id": row[3],
                    "words": row[4],
                    "shards": row[5],
                }
                for row in cur.fetchall()
            ]
    typer.echo(json.dumps(rows, indent=2))


__all__ = [
    "sources_catalog",
    "sources_preflight",
    "sources_status",
]
