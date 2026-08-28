"""Root Typer CLI interface for F2 study suites, corpus pipeline, and reproduction catalog."""

from __future__ import annotations

import typer

from .catalog.cli import app as catalog_app
from .corpus.cli import app as corpus_app

app = typer.Typer(
    name="f2",
    help="Word2Vec (2013) Paper Reproduction, Corpus Pipeline & Catalog.",
    no_args_is_help=True,
)

app.add_typer(corpus_app, name="corpus")
app.add_typer(catalog_app, name="catalog")

__all__ = ["app"]
