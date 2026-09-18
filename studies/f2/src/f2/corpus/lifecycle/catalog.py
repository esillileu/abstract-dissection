"""Idempotent cataloging of resource versions and validation profiles."""

from __future__ import annotations

import json
from typing import Any

from ..db.repository import CorpusStateRepository
from ..sources import SOURCES, VALIDATION_PROFILES
from .preflight import config_hash


def catalog_sources(conn: Any) -> None:
    """Idempotently register source, canonical, and normalized resource versions."""
    with conn.cursor() as cur:
        for source in SOURCES:
            status = "blocked" if source.blocked_reason else "pending"
            for suffix, name, access in (
                ("raw", f"{source.name} raw release", source.access),
                ("canonical", f"{source.name} canonical text", source.access),
                (
                    "normalized",
                    f"{source.name} word2vec-normalized text",
                    source.access,
                ),
            ):
                resource_id = f"f2-{source.key}-{suffix}"
                version_id = getattr(source, f"{suffix}_resource_version_id")
                cur.execute(
                    """INSERT INTO catalog.resources
                       (resource_id, kind, name, description, access_status, acquisition_status, readiness_status, notes)
                       VALUES (%s, 'dataset', %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (resource_id) DO UPDATE SET name=EXCLUDED.name, description=EXCLUDED.description,
                         access_status=EXCLUDED.access_status, acquisition_status=EXCLUDED.acquisition_status,
                         readiness_status=EXCLUDED.readiness_status, notes=EXCLUDED.notes""",
                    (
                        resource_id,
                        name,
                        f"Frozen F2 source {source.key} release {source.release}",
                        access,
                        status,
                        status,
                        source.blocked_reason,
                    ),
                )
                metadata = {
                    "source_key": source.key,
                    "representation": suffix,
                    "release": source.release,
                    "source_config_hash": source.config_hash,
                }
                cur.execute(
                    """INSERT INTO catalog.resource_versions
                       (resource_version_id, resource_id, version_label, uri, metadata, is_verified)
                       VALUES (%s, %s, %s, %s, %s, FALSE)
                       ON CONFLICT (resource_version_id) DO NOTHING""",
                    (
                        version_id,
                        resource_id,
                        source.release if suffix == "raw" else "v1",
                        source.homepage if suffix == "raw" else None,
                        json.dumps(metadata),
                    ),
                )
            cur.execute(
                """INSERT INTO catalog.resource_sources (resource_id, source_type, url, license, is_preferred, notes)
                   SELECT %s, 'official_release', %s, %s, TRUE, %s
                   WHERE NOT EXISTS (SELECT 1 FROM catalog.resource_sources WHERE resource_id=%s AND is_preferred=TRUE)""",
                (
                    f"f2-{source.key}-raw",
                    source.homepage,
                    source.license,
                    source.blocked_reason,
                    f"f2-{source.key}-raw",
                ),
            )
    conn.commit()


def install_validation_profiles(repo: CorpusStateRepository) -> None:
    for key, spec in VALIDATION_PROFILES.items():
        profile_id = key
        if repo.get_validation_profile(profile_id):
            continue
        repo.create_validation_profile(
            profile_id,
            key.rsplit("-v", 1)[0],
            spec["revision"],
            key,
            config_hash(spec),
            spec,
        )


__all__ = ["catalog_sources", "install_validation_profiles"]
