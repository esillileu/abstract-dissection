"""F2 corpus release acquisition and preprocessing commands.

Common Crawl sampling and audit commands belong exclusively to the f2-cc study.
"""

from __future__ import annotations

import typer

from .cli import fineweb_app, sources_app

app = typer.Typer(
    name="corpus",
    help="Acquire, preprocess, validate, and register completed corpus releases.",
    no_args_is_help=True,
)
app.add_typer(sources_app, name="sources")
app.add_typer(fineweb_app, name="fineweb")

__all__ = ["app"]
