"""Typer CLI for the repository's local MLflow service and file transfers."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from .transfer import export_experiment, export_run, import_archive

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPORT_DIRECTORY = Path("infra/mlflow/exports")
DEFAULT_COMPOSE_FILE = Path("infra/mlflow/compose.yaml")

app = typer.Typer(
    no_args_is_help=True,
    help="Manage local MLflow and transfer experiments through ZIP archives.",
)


def _tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def _archive_path(name: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not safe_name:
        raise typer.BadParameter("experiment/run name cannot produce an archive name")
    return DEFAULT_EXPORT_DIRECTORY / f"{safe_name}.zip"


def _destination_tags(values: list[str] | None) -> dict[str, str]:
    tags: dict[str, str] = {}
    for value in values or []:
        key, separator, tag_value = value.partition("=")
        if not separator or not key:
            raise typer.BadParameter(
                f"destination tag must be KEY=VALUE: {value}",
                param_hint="--destination-tag",
            )
        tags[key] = tag_value
    return tags


def _compose(*arguments: str) -> None:
    compose_file = Path(
        os.getenv("MLFLOW_COMPOSE_FILE", str(DEFAULT_COMPOSE_FILE))
    )
    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), *arguments],
        check=True,
    )


@app.command()
def up() -> None:
    """Start the local MLflow server."""
    _compose("up", "-d")


@app.command()
def down() -> None:
    """Stop the local MLflow server without deleting its data."""
    _compose("down")


@app.command()
def logs(
    follow: Annotated[
        bool, typer.Option("--follow", "-f", help="Follow log output.")
    ] = False,
    tail: Annotated[
        str, typer.Option("--tail", help="Number of lines or 'all'.")
    ] = "all",
) -> None:
    """Show local MLflow server logs."""
    arguments = ["logs", "--tail", tail]
    if follow:
        arguments.append("--follow")
    arguments.append("mlflow")
    _compose(*arguments)


@app.command("export")
def export_experiment_command(
    experiment: Annotated[str, typer.Argument(help="Experiment name or numeric ID.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Destination ZIP path."),
    ] = None,
    tracking_uri: Annotated[
        str | None,
        typer.Option(
            "--tracking-uri",
            help="Source tracking URI; defaults to MLFLOW_TRACKING_URI.",
        ),
    ] = None,
) -> None:
    """Export FINISHED runs from an experiment."""
    archive = export_experiment(
        tracking_uri or _tracking_uri(),
        experiment,
        output or _archive_path(experiment),
    )
    typer.echo(json.dumps({"archive": str(archive)}, ensure_ascii=False))


@app.command("export-run")
def export_run_command(
    run_id: Annotated[str, typer.Argument(help="Run ID to export.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Destination ZIP path."),
    ] = None,
    tracking_uri: Annotated[
        str | None,
        typer.Option(
            "--tracking-uri",
            help="Source tracking URI; defaults to MLFLOW_TRACKING_URI.",
        ),
    ] = None,
) -> None:
    """Export one explicitly selected run regardless of status."""
    archive = export_run(
        tracking_uri or _tracking_uri(),
        run_id,
        output or _archive_path(f"run-{run_id}"),
    )
    typer.echo(json.dumps({"archive": str(archive)}, ensure_ascii=False))


@app.command("import")
def import_experiment_command(
    experiment: Annotated[
        str,
        typer.Argument(
            help="Target experiment name and default archive filename."
        ),
    ],
    input_path: Annotated[
        Path | None,
        typer.Option("--input", "-i", help="Source ZIP path."),
    ] = None,
    tracking_uri: Annotated[
        str | None,
        typer.Option(
            "--tracking-uri",
            help="Target tracking URI; defaults to MLFLOW_TRACKING_URI.",
        ),
    ] = None,
    reuse_experiment: Annotated[
        bool,
        typer.Option(
            "--reuse-experiment/--no-reuse-experiment",
            help="Append to an existing experiment with the same name.",
        ),
    ] = True,
    environment_tags: Annotated[
        bool,
        typer.Option(
            "--environment-tags/--no-environment-tags",
            help="Record hardware tags from the machine running this command.",
        ),
    ] = True,
    destination_tag: Annotated[
        list[str] | None,
        typer.Option(
            "--destination-tag",
            help="Repeatable KEY=VALUE tag for the experiment and imported runs.",
        ),
    ] = None,
    artifact_location: Annotated[
        str | None,
        typer.Option(
            "--artifact-location",
            help="Artifact location used only when creating an experiment.",
        ),
    ] = None,
) -> None:
    """Import an archive, appending to the named experiment by default."""
    result = import_archive(
        tracking_uri or _tracking_uri(),
        input_path or _archive_path(experiment),
        experiment_name=experiment,
        artifact_location=artifact_location,
        capture_environment=environment_tags,
        destination_tags=_destination_tags(destination_tag),
        reuse_experiment=reuse_experiment,
    )
    typer.echo(json.dumps(result, ensure_ascii=False))
