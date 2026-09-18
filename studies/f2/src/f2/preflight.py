"""Read-only identity checks for F2 experiment services."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit

import psycopg

from .corpus.db.contract import (
    database_name,
    validate_connection,
    validate_database_url,
)
from .corpus.object_store import s3_config_from_environment
from .tracking import resolve_tracking_uri


class PreflightError(RuntimeError):
    """Raised before an F2 run can access a misidentified service."""


@dataclass(frozen=True)
class ServiceIdentity:
    service: str
    scheme: str
    host: str
    name: str


def _network_identity(service: str, url: str, *, name: str = "") -> ServiceIdentity:
    parsed = urlsplit(url)
    if not parsed.scheme or not parsed.hostname:
        raise PreflightError(f"{service} endpoint must be an absolute URL")
    return ServiceIdentity(service, parsed.scheme, parsed.hostname, name)


def database_identity(connection_url: str | None = None) -> ServiceIdentity:
    """Verify the canonical DB and schemas using a read-only transaction."""
    url = connection_url or os.getenv("F2_DATABASE_URL", "").strip()
    if not url:
        raise PreflightError("F2_DATABASE_URL is required")
    try:
        validate_database_url(url)
        with psycopg.connect(
            url, options="-c default_transaction_read_only=on"
        ) as conn:
            validate_connection(conn, schema="catalog")
    except Exception as exc:
        raise PreflightError(str(exc)) from None
    parsed = urlsplit(url)
    return ServiceIdentity(
        "postgresql", parsed.scheme, parsed.hostname or "", database_name(url)
    )


def corpus_store_identity(*, probe: bool = True) -> ServiceIdentity:
    """Validate a local S3 endpoint or worker-facing Tailscale HTTPS endpoint."""
    config = s3_config_from_environment()
    identity = _network_identity("corpus-s3", config.endpoint, name=config.root_uri)
    parsed = urlsplit(config.endpoint)
    is_local = (
        identity.scheme == "http"
        and identity.host in {"127.0.0.1", "localhost", "::1"}
        and parsed.port in {9000, 19000}
    )
    is_worker = identity.scheme == "https" and identity.host.endswith(".ts.net")
    if not (is_local or is_worker):
        raise PreflightError(
            "F2_CORPUS_S3_ENDPOINT must be local HTTP on port 9000/19000 or a "
            "Tailscale HTTPS endpoint (*.ts.net)"
        )
    if probe:
        from repro_io.s3 import S3ObjectStore

        try:
            S3ObjectStore(config).probe()
        except Exception as exc:
            raise PreflightError(
                f"corpus-s3 read-only probe failed: {type(exc).__name__}"
            ) from None
    return identity


def mlflow_identity(
    tracking_uri: str | None = None, *, probe: bool = True
) -> ServiceIdentity:
    """Validate an HTTP(S) tracking endpoint with a read-only API request."""
    uri = resolve_tracking_uri(tracking_uri)
    identity = _network_identity("mlflow", uri)
    if identity.scheme not in {"http", "https"}:
        raise PreflightError("F2 MLflow tracking URI must use HTTP(S)")
    if probe:
        try:
            from mlflow import MlflowClient

            MlflowClient(tracking_uri=uri).search_experiments(max_results=1)
        except Exception as exc:
            raise PreflightError(
                f"MLflow read-only probe failed: {type(exc).__name__}"
            ) from None
    return identity


def run_preflight(*, probe: bool = True) -> dict[str, dict[str, Any]]:
    """Return only redacted service identities; never URLs or credentials."""
    return {
        "database": asdict(database_identity()),
        "corpus_store": asdict(corpus_store_identity(probe=probe)),
        "mlflow": asdict(mlflow_identity(probe=probe)),
    }


__all__ = [
    "PreflightError",
    "ServiceIdentity",
    "corpus_store_identity",
    "database_identity",
    "mlflow_identity",
    "run_preflight",
]
