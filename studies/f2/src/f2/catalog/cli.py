"""Typer CLI interface for F2 reproduction catalog DB and plan management."""

from __future__ import annotations

import typer

from .db.migrations.runner import run_catalog_migrations
from .db.repository import CatalogRepository
from .db.session import get_connection

app = typer.Typer(
    name="catalog",
    help="Manage F2 reproduction catalog, resource inventory, and canonical execution plans.",
    no_args_is_help=True,
)


@app.command("migrate")
def migrate_catalog_db() -> None:
    """Apply pending PostgreSQL migrations for the F2 reproduction catalog database."""
    with get_connection() as conn:
        applied = run_catalog_migrations(conn)
        if applied:
            typer.echo(
                f"Successfully applied {len(applied)} catalog migrations: {', '.join(applied)}"
            )
        else:
            typer.echo("Catalog database schema is up to date (0 pending migrations).")


@app.command("status")
def show_status(
    plan_key: str = typer.Option("f2", "--plan-key", "-k", help="Plan key identifier"),
) -> None:
    """Display canonical execution plan progress and resource inventory."""
    with get_connection() as conn:
        repo = CatalogRepository(conn)
        progress = repo.get_canonical_plan_progress(plan_key=plan_key)
        inventory = repo.get_resource_inventory()

    typer.echo(f"=== Canonical Plan Progress ({plan_key}) ===")
    if not progress:
        typer.echo(f"No canonical execution plan found for key '{plan_key}'.")
    else:
        for p in progress:
            typer.echo(
                f"  - Exp [{p['experiment_spec_id']}] {p['experiment_name']}: "
                f"{p['executed_slots']}/{p['expected_slots']} executed "
                f"({p['completion_rate'] * 100:.1f}%), {p['missing_slots']} missing"
            )

    typer.echo("\n=== Resource Inventory ===")
    if not inventory:
        typer.echo("No resources registered in catalog.")
    else:
        for r in inventory:
            typer.echo(
                f"  - [{r['kind']}] {r['name']} ({r['resource_id']}): "
                f"access={r['access_status']}, acquisition={r['acquisition_status']}, readiness={r['readiness_status']}"
            )


@app.command("matrix")
def show_matrix(
    plan_key: str = typer.Option("f2", "--plan-key", "-k", help="Plan key identifier"),
) -> None:
    """Display canonical run matrix with MLflow execution pointers."""
    with get_connection() as conn:
        repo = CatalogRepository(conn)
        matrix = repo.get_canonical_run_matrix(plan_key=plan_key)

    typer.echo(f"=== Canonical Run Matrix ({plan_key}) ===")
    if not matrix:
        typer.echo(f"No slots found for canonical plan '{plan_key}'.")
    else:
        for row in matrix:
            status_str = "EXECUTED" if row["executed"] else "MISSING"
            mlflow_ptr = row["reference_mlflow_run_id"] or "none"
            typer.echo(
                f"  - [{status_str}] {row['experiment_spec_id']} / {row['slot_key']} (mlflow_id={mlflow_ptr})"
            )


__all__ = ["app"]
