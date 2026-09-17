"""Composite multi-stage runner CLI command for corpus lifecycle."""

from __future__ import annotations

from typing import Annotated

import typer

from .acquire import sources_acquire
from .catalog import sources_catalog
from .process import sources_process
from .validate import sources_validate


def sources_run(
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
    """Run acquire; processing and validation remain individually resumable commands."""
    sources_catalog()
    sources_acquire(
        source=source,
        checksum=checksum,
        peak_mbps=peak_mbps,
        offpeak_mbps=offpeak_mbps,
    )
    sources_process(source=source, target_words=10_000_000)
    sources_validate(source=source)


__all__ = ["sources_run"]
