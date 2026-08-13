"""Unified Typer CLI for declared experiment domains."""

from __future__ import annotations

from collections.abc import Sequence

import typer

from exp.deepscratch.definition import DEFINITION
from exp.framework import CommandGroups, DomainRegistry


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
storage_app = typer.Typer(help="Audit and clean experiment result storage.")
app.add_typer(plan_app, name="plan")
app.add_typer(run_app, name="run")
app.add_typer(analyze_app, name="analyze")
app.add_typer(check_app, name="check")
app.add_typer(profile_app, name="profile")
app.add_typer(storage_app, name="storage")


PLUGIN_REGISTRY = DomainRegistry()
PLUGIN_REGISTRY.register(
    DEFINITION,
    CommandGroups(
        root=app,
        plan=plan_app,
        run=run_app,
        analyze=analyze_app,
        check=check_app,
        profile=profile_app,
        storage=storage_app,
    ),
)


def main(argv: Sequence[str] | None = None) -> None:
    app(args=None if argv is None else list(argv), prog_name="exp")
