"""Resources, resource versions, sources, and preparation specs operations for F2 Catalog DB."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseCatalogRepository


class ResourcesRepositoryMixin(BaseCatalogRepository):
    """Transactional operations for catalog resources and preparation specs."""

    # 4. Resources & Versions
    def upsert_resource(
        self,
        resource_id: str,
        kind: str,
        name: str,
        access_status: str,
        acquisition_status: str,
        readiness_status: str,
        description: str | None = None,
        canonical_version_id: str | None = None,
        notes: str | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO resources (
                    resource_id, kind, name, description, access_status,
                    acquisition_status, readiness_status, canonical_version_id, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (resource_id) DO UPDATE SET
                    kind = EXCLUDED.kind,
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    access_status = EXCLUDED.access_status,
                    acquisition_status = EXCLUDED.acquisition_status,
                    readiness_status = EXCLUDED.readiness_status,
                    canonical_version_id = EXCLUDED.canonical_version_id,
                    notes = EXCLUDED.notes;
                """,
                (
                    resource_id,
                    kind,
                    name,
                    description,
                    access_status,
                    acquisition_status,
                    readiness_status,
                    canonical_version_id,
                    notes,
                ),
            )

    def set_resource_canonical_version(
        self,
        resource_id: str,
        resource_version_id: str | None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE resources
                SET canonical_version_id = %s
                WHERE resource_id = %s;
                """,
                (resource_version_id, resource_id),
            )

    def upsert_resource_version(
        self,
        resource_version_id: str,
        resource_id: str,
        version_label: str | None = None,
        uri: str | None = None,
        local_path: str | None = None,
        checksum_algo: str | None = None,
        checksum: str | None = None,
        size_bytes: int | None = None,
        metadata: dict[str, Any] | None = None,
        is_verified: bool = False,
        notes: str | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO resource_versions (
                    resource_version_id, resource_id, version_label, uri, local_path,
                    checksum_algo, checksum, size_bytes, metadata, is_verified, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (resource_version_id) DO UPDATE SET
                    resource_id = EXCLUDED.resource_id,
                    version_label = EXCLUDED.version_label,
                    uri = EXCLUDED.uri,
                    local_path = EXCLUDED.local_path,
                    checksum_algo = EXCLUDED.checksum_algo,
                    checksum = EXCLUDED.checksum,
                    size_bytes = EXCLUDED.size_bytes,
                    metadata = EXCLUDED.metadata,
                    is_verified = EXCLUDED.is_verified,
                    notes = EXCLUDED.notes;
                """,
                (
                    resource_version_id,
                    resource_id,
                    version_label,
                    uri,
                    local_path,
                    checksum_algo,
                    checksum,
                    size_bytes,
                    json.dumps(metadata or {}),
                    is_verified,
                    notes,
                ),
            )

    def upsert_resource_source(
        self,
        resource_id: str,
        source_type: str,
        url: str | None = None,
        citation: str | None = None,
        license: str | None = None,
        is_preferred: bool = False,
        notes: str | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            if is_preferred:
                # Clear previous preferred source for this resource if set
                cur.execute(
                    """
                    UPDATE resource_sources
                    SET is_preferred = FALSE
                    WHERE resource_id = %s AND is_preferred = TRUE;
                    """,
                    (resource_id,),
                )
            cur.execute(
                """
                INSERT INTO resource_sources (
                    resource_id, source_type, url, citation, license, is_preferred, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    resource_id,
                    source_type,
                    url,
                    citation,
                    license,
                    is_preferred,
                    notes,
                ),
            )

    def upsert_preparation_spec(
        self,
        preparation_id: str,
        input_resource_id: str,
        preparation_type: str,
        specification: dict[str, Any],
        output_resource_id: str | None = None,
        code_resource_id: str | None = None,
        status: str = "planned",
        notes: str | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO preparation_specs (
                    preparation_id, input_resource_id, output_resource_id,
                    preparation_type, specification, code_resource_id, status, notes
                ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                ON CONFLICT (preparation_id) DO UPDATE SET
                    input_resource_id = EXCLUDED.input_resource_id,
                    output_resource_id = EXCLUDED.output_resource_id,
                    preparation_type = EXCLUDED.preparation_type,
                    specification = EXCLUDED.specification,
                    code_resource_id = EXCLUDED.code_resource_id,
                    status = EXCLUDED.status,
                    notes = EXCLUDED.notes;
                """,
                (
                    preparation_id,
                    input_resource_id,
                    output_resource_id,
                    preparation_type,
                    json.dumps(specification),
                    code_resource_id,
                    status,
                    notes,
                ),
            )
