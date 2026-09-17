"""Typer CLI for acquiring and preprocessing completed corpus releases."""

from __future__ import annotations

import typer

from .acquire import import_gigaword, sources_acquire
from .catalog import sources_catalog, sources_preflight, sources_status
from .common import (
    _BW_OFFPEAK_DEFAULT,
    _BW_PEAK_DEFAULT,
    _bandwidth,
    _selected,
    _store,
)
from .orchestration import sources_run
from .process import _canonical_inputs, _raw_inputs, sources_process
from .validate import sources_validate

app = typer.Typer(
    name="corpus",
    help="Acquire and preprocess completed corpus releases.",
    no_args_is_help=True,
)
sources_app = typer.Typer(
    name="sources",
    help="Canonical public and licensed corpus lifecycle.",
    no_args_is_help=True,
)
app.add_typer(sources_app, name="sources")

sources_app.command("preflight")(sources_preflight)
sources_app.command("catalog")(sources_catalog)
sources_app.command("acquire")(sources_acquire)
sources_app.command("process")(sources_process)
sources_app.command("validate")(sources_validate)
sources_app.command("status")(sources_status)
sources_app.command("import-gigaword")(import_gigaword)
sources_app.command("run")(sources_run)

__all__ = [
    "_BW_OFFPEAK_DEFAULT",
    "_BW_PEAK_DEFAULT",
    "_bandwidth",
    "_canonical_inputs",
    "_raw_inputs",
    "_selected",
    "_store",
    "app",
    "import_gigaword",
    "sources_acquire",
    "sources_app",
    "sources_catalog",
    "sources_preflight",
    "sources_process",
    "sources_run",
    "sources_status",
    "sources_validate",
]
