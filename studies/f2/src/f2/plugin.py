"""F2 (Word2Vec 2013) Study Plugin Definition and Repro CLI registration."""

from __future__ import annotations

from dataclasses import dataclass

import typer

from repro_core.registry import CommandGroups

from . import cli
from .catalog import cli as catalog_cli


@dataclass(frozen=True)
class F2StudyPlugin:
    name: str = "f2"
    display_name: str = "Word2Vec (2013) Paper Reproduction"
    description: str = "Common Crawl (2009-2012) corpus feasibility and Word2Vec models"

    def register_commands(self, groups: CommandGroups) -> None:
        """Attach study-owned commands to root CLI groups."""
        # Top-level commands
        groups.plan.add_typer(cli.app, name=self.name)
        groups.run.add_typer(cli.app, name=self.name)
        groups.analyze.add_typer(cli.app, name=self.name)

        # Study-first subcommand: repro f2 ...
        f2_app = typer.Typer(
            name=self.name,
            help="Word2Vec 2013 reproduction, Common Crawl corpus pipeline & catalog.",
            no_args_is_help=True,
        )
        f2_app.add_typer(cli.app, name="corpus")
        f2_app.add_typer(catalog_cli.app, name="catalog")

        groups.root.add_typer(f2_app, name=self.name)


PLUGIN = F2StudyPlugin()
DEFINITION = PLUGIN

__all__ = ["DEFINITION", "PLUGIN", "F2StudyPlugin"]
