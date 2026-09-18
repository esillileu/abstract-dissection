"""CLI commands for external corpus source acquisition and licensed data import."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from repro_io.checksum import sha256_file

from repro_core.context.paths import RuntimePaths

from ..db.repository import CorpusStateRepository
from ..db.session import get_connection
from ..lifecycle import acquire_source
from .common import _bandwidth, _selected, _store


def sources_acquire(
    source: Annotated[str, typer.Option("--source", "-s")] = "all",
    checksum: Annotated[
        list[str] | None,
        typer.Option("--checksum", help="NAME=SHA256; required for LM1B and Wikipedia"),
    ] = None,
    peak_mbps: Annotated[
        float | None,
        typer.Option(
            "--peak-mbps",
            help="Download rate limit during peak hours 09:00-22:00 (default: 40 Mbps)",
        ),
    ] = None,
    offpeak_mbps: Annotated[
        float | None,
        typer.Option(
            "--offpeak-mbps",
            help="Download rate limit during off-peak hours 22:00-09:00 (default: 100 Mbps)",
        ),
    ] = None,
) -> None:
    """Download, upload, remotely verify, and transactionally register raw releases."""
    overrides: dict[str, str] = {}
    for item in checksum or []:
        name, separator, digest = item.partition("=")
        if not separator or len(digest) != 64:
            raise typer.BadParameter("--checksum must be NAME=64_HEX_SHA256")
        overrides[name] = digest.lower()
    paths, store = RuntimePaths.from_environment(), _store()
    bw = _bandwidth(peak_mbps, offpeak_mbps)
    with get_connection() as conn:
        repo = CorpusStateRepository(conn)
        for spec in _selected(source):
            try:
                artifacts = acquire_source(
                    spec,
                    paths.staging_root / "exp" / "f2" / "sources",
                    store,
                    repo,
                    checksum_overrides=overrides,
                    bandwidth=bw,
                )
                typer.echo(f"{spec.key}: ACQUIRED ({len(artifacts)} artifacts)")
            except PermissionError as exc:
                typer.echo(f"{spec.key}: BLOCKED: {exc}")
            except Exception as exc:
                typer.echo(f"{spec.key}: FAILED: {exc}", err=True)


def import_gigaword(
    archive: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Import a licensed archive using the separate restricted S3 credential/root."""
    store = _store(restricted=True)
    digest = sha256_file(archive)
    uri = store.uri(f"raw/gigaword/LDC2011T07/{archive.name}")
    store.put_file(archive, uri)
    if store.sha256(uri) != digest:
        raise typer.Exit(1)
    typer.echo(f"verified restricted import: {uri} sha256={digest}")


__all__ = [
    "import_gigaword",
    "sources_acquire",
]
