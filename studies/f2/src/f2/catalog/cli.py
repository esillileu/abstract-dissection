"""Typer CLI interface for F2 research catalog manifest inspection."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .manifest import digest_manifest, read_manifest

app = typer.Typer(
    name="catalog",
    help="Inspect and validate F2 research catalog manifest.",
    no_args_is_help=True,
)


@app.command("validate")
def validate_catalog_manifest(
    manifest: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    """Validate a research catalog manifest file without database access."""
    payload = read_manifest(manifest)
    digest = digest_manifest(payload)
    typer.echo(
        json.dumps(
            {
                "valid": True,
                "sha256": digest,
                "sections": {
                    k: len(v) for k, v in payload.items() if isinstance(v, list)
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


__all__ = ["app"]
