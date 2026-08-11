"""Unified Typer CLI for declared experiment domains."""

from __future__ import annotations

from collections.abc import Sequence

import typer

from exp.domain import DomainDefinition
from exp.ds1 import cli as ds1_cli
from exp.ds2 import cli as ds2_cli
from exp.ds1_original import cli as ds1_original_cli
from exp.ds2_original import cli as ds2_original_cli


app = typer.Typer(
    name="exp",
    help="Plan, run, analyze, and profile declared experiment domains.",
    no_args_is_help=True,
)
plan_app = typer.Typer(help="Inspect expanded experiment run plans.")
run_app = typer.Typer(help="Execute catalog or original-source experiments.")
analyze_app = typer.Typer(help="Render or summarize experiment results.")
profile_app = typer.Typer(help="Profile experiment update and module runtimes.")
app.add_typer(plan_app, name="plan")
app.add_typer(run_app, name="run")
app.add_typer(analyze_app, name="analyze")
app.add_typer(profile_app, name="profile")


DOMAIN_REGISTRY: dict[str, DomainDefinition] = {}


def _register_domain(name: str, module: object) -> None:
    definition = getattr(module, "DEFINITION")
    if name in DOMAIN_REGISTRY:
        raise RuntimeError(f"duplicate experiment domain: {name}")
    if definition.name != name:
        raise RuntimeError(
            f"domain registry name mismatch: {name} != {definition.name}"
        )
    DOMAIN_REGISTRY[name] = definition
    plan_app.command(name)(getattr(module, "plan"))
    run_app.command(name)(getattr(module, "run"))
    analyze_app.command(name)(getattr(module, "analyze"))
    profile = getattr(module, "profile", None)
    if profile is not None:
        profile_app.command(name)(profile)


_register_domain("ds1", ds1_cli)
_register_domain("ds2", ds2_cli)
_register_domain("ds1_original", ds1_original_cli)
_register_domain("ds2_original", ds2_original_cli)


def main(argv: Sequence[str] | None = None) -> None:
    app(args=None if argv is None else list(argv), prog_name="exp")
