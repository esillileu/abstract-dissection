"""Unified Typer CLI for declared reproduction study domains."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Sequence

import typer

from ..context.paths import RuntimePaths
from ..registry.plugin import CommandGroups, DomainRegistry

app = typer.Typer(
    name="repro",
    help="Plan, run, analyze, and profile paper reproduction studies.",
    no_args_is_help=True,
)
plan_app = typer.Typer(help="Inspect expanded experiment run plans.")
run_app = typer.Typer(help="Execute catalog or reference experiments.")
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

COMMAND_GROUPS = CommandGroups(
    root=app,
    plan=plan_app,
    run=run_app,
    analyze=analyze_app,
    check=check_app,
    profile=profile_app,
    storage=storage_app,
)

PLUGIN_REGISTRY = DomainRegistry()


def discover_and_register_plugins() -> None:
    """Discover study plugins via entry points or direct imports."""
    # 1. Entry points
    try:
        eps = importlib.metadata.entry_points(group="repro.studies")
        for ep in eps:
            try:
                plugin = ep.load()
                if plugin.name not in PLUGIN_REGISTRY.names():
                    PLUGIN_REGISTRY.register(plugin, COMMAND_GROUPS)
            except Exception:
                pass
    except Exception:
        pass

    # 2. Fallback import for dlfs
    if "dlfs" not in PLUGIN_REGISTRY.names():
        try:
            DLFS_PLUGIN = importlib.import_module("dlfs.plugin").PLUGIN
            PLUGIN_REGISTRY.register(DLFS_PLUGIN, COMMAND_GROUPS)
        except Exception:
            pass


@app.command("list")
def list_studies() -> None:
    """List all registered reproduction studies."""
    discover_and_register_plugins()
    studies = PLUGIN_REGISTRY.names()
    if not studies:
        typer.echo("No reproduction studies registered.")
        return
    typer.echo("Registered reproduction studies:")
    for name in sorted(studies):
        typer.echo(f"  - {name}")


@app.command("info")
def info() -> None:
    """Show repository runtime paths and environment information."""
    paths = RuntimePaths.from_environment()
    typer.echo(f"Repository Root: {paths.repo_root}")
    typer.echo(f"Data Root:       {paths.data_root}")
    typer.echo(f"Artifacts Root:  {paths.artifacts_root}")
    typer.echo(f"Cache Root:      {paths.cache_root}")
    typer.echo(f"References Root: {paths.references_root}")
    typer.echo(f"Studies Root:    {paths.studies_root}")

    try:
        from ..numerics import BackendConfig, make_backend

        backend = make_backend(BackendConfig(device="cpu"))
        typer.echo(f"Default Backend: {backend.name} ({backend.device})")
    except Exception as exc:
        typer.echo(f"Default Backend Error: {exc}")


# Auto-discover at module load time for tests and sub-apps
discover_and_register_plugins()


def main(argv: Sequence[str] | None = None) -> None:
    discover_and_register_plugins()
    app(args=None if argv is None else list(argv), prog_name="repro")


if __name__ == "__main__":
    main()
