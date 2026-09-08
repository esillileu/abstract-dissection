"""F2 (Word2Vec 2013) Study Plugin Definition and Repro CLI registration."""

from __future__ import annotations

from dataclasses import dataclass

from repro_core.registry import CommandGroups

from .definition import DEFINITION, F2Definition


@dataclass(frozen=True)
class F2StudyPlugin:
    """Entry point plugin registering F2 subcommands into repro CLI."""

    name: str = "f2"
    display_name: str = "Word2Vec (2013) Paper Reproduction Campaign"
    description: str = "Common Crawl (2009-2012) corpus feasibility, word embedding reproductions, and evaluation suites"

    def register_commands(self, groups: CommandGroups) -> None:
        """Attach domain-owned commands to root CLI groups."""
        from . import cli

        # Verb-first commands: repro plan/run/analyze/check f2 <suite> ...
        groups.plan.command(
            self.name,
            help="Inspect expanded experiment run plans for an F2 suite.",
        )(cli.plan)
        groups.run.command(self.name, help="Execute F2 suite catalog experiments.")(
            cli.run
        )
        groups.analyze.command(
            self.name, help="Render or summarize F2 experiment results."
        )(cli.analyze)
        groups.check.command(
            self.name,
            help="Compare declared plans with recorded F2 run state.",
        )(cli.check)

        # Study-first subcommand: repro f2 [corpus|catalog|suites] ...
        groups.root.add_typer(cli.app, name=self.name)


PLUGIN = F2StudyPlugin()

__all__ = ["DEFINITION", "PLUGIN", "F2Definition", "F2StudyPlugin"]
