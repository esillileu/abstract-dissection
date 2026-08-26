"""DLFS (Deep Learning from Scratch) Study Plugin Definition."""

from __future__ import annotations

from dataclasses import dataclass

import typer

from repro_core.execution import ExecutionDefinition
from repro_core.registry import CommandGroups

from .ds1.catalog import IMPLEMENTED as DS1_IMPLEMENTED
from .ds1.catalog import ORIGINAL as DS1_ORIGINAL
from .ds2.catalog import IMPLEMENTED as DS2_IMPLEMENTED
from .ds2.catalog import ORIGINAL as DS2_ORIGINAL
from .identity import Variant, Volume


@dataclass(frozen=True)
class DLFSStudyPlugin:
    name: str = "dlfs"
    display_name: str = "Deep Learning from Scratch"
    description: str = (
        "Reproductions and analyses for Deep Learning from Scratch Volumes 1 & 2"
    )

    def implementation(self, volume: Volume, variant: Variant) -> ExecutionDefinition:
        return {
            (Volume.DS1, Variant.IMPLEMENTED): DS1_IMPLEMENTED,
            (Volume.DS1, Variant.ORIGINAL): DS1_ORIGINAL,
            (Volume.DS2, Variant.IMPLEMENTED): DS2_IMPLEMENTED,
            (Volume.DS2, Variant.ORIGINAL): DS2_ORIGINAL,
        }[(volume, variant)]

    def register_commands(self, groups: CommandGroups) -> None:
        """Attach domain-owned commands to root CLI groups."""
        from . import cli

        for alias in (self.name, "deepscratch"):
            groups.plan.command(alias, help="Inspect DLFS experiment run plans.")(
                cli.plan
            )
            groups.run.command(
                alias, help="Execute DLFS catalog or reference experiments."
            )(cli.run)
            groups.analyze.command(
                alias, help="Render or summarize DLFS experiment results."
            )(cli.analyze)
            groups.check.command(
                alias, help="Compare declared plans with recorded DLFS run state."
            )(cli.check)
            groups.profile.command(
                alias, help="Profile DLFS update and module runtimes."
            )(cli.profile)

        # Support study-first subcommand: repro dlfs ...
        dlfs_app = typer.Typer(
            name=self.name,
            help="Deep Learning from Scratch (Vol 1 & 2) reproduction study.",
            no_args_is_help=True,
        )
        dlfs_app.command("plan", help="Inspect expanded experiment run plans.")(
            cli.plan
        )
        dlfs_app.command("run", help="Execute catalog or reference experiments.")(
            cli.run
        )
        dlfs_app.command("analyze", help="Render or summarize experiment results.")(
            cli.analyze
        )
        dlfs_app.command(
            "check", help="Compare declared plans with recorded run state."
        )(cli.check)
        dlfs_app.command(
            "profile", help="Profile experiment update and module runtimes."
        )(cli.profile)

        groups.root.add_typer(dlfs_app, name=self.name)


PLUGIN = DLFSStudyPlugin()
DEFINITION = PLUGIN
