"""Unified Typer CLI for declared experiment domains."""

from __future__ import annotations

from collections.abc import Sequence

import typer

from exp.deepscratch import cli as deepscratch_cli
from exp.framework import DomainRegistry


app = typer.Typer(
    name="exp",
    help="Plan, run, analyze, and profile declared experiment domains.",
    no_args_is_help=True,
)
plan_app = typer.Typer(help="Inspect expanded experiment run plans.")
run_app = typer.Typer(help="Execute catalog or original-source experiments.")
analyze_app = typer.Typer(help="Render or summarize experiment results.")
check_app = typer.Typer(help="Compare declared plans with recorded run state.")
profile_app = typer.Typer(help="Profile experiment update and module runtimes.")
app.add_typer(plan_app, name="plan")
app.add_typer(run_app, name="run")
app.add_typer(analyze_app, name="analyze")
app.add_typer(check_app, name="check")
app.add_typer(profile_app, name="profile")


PLUGIN_REGISTRY = DomainRegistry()
PLUGIN_REGISTRY.register(deepscratch_cli.DEFINITION)
plan_app.command("deepscratch")(deepscratch_cli.plan)
run_app.command("deepscratch")(deepscratch_cli.run)
analyze_app.command("deepscratch")(deepscratch_cli.analyze)
check_app.command("deepscratch")(deepscratch_cli.check)
profile_app.command("deepscratch")(deepscratch_cli.profile)
app.command("import-legacy")(deepscratch_cli.import_legacy)


def main(argv: Sequence[str] | None = None) -> None:
    app(args=None if argv is None else list(argv), prog_name="exp")
