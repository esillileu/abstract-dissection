"""Run command implementation executing F2 experiment runs."""

from __future__ import annotations

from repro_core.execution.definition import RunOrder

from ..execution.definitions import DEFINITION
from ..suites.common.w2v.tracked import run_tracked_yaml
from ..tracking import resolve_tracking_uri


def run_command_impl(
    suite: str,
    *,
    experiment: list[str] | None = None,
    all_experiments: bool = False,
    atomic_run: list[str] | None = None,
    exclude_atomic_run: list[str] | None = None,
    seed_set: str | None = None,
    seed: str | None = None,
    device: str | None = None,
    override_values: list[str] | None = None,
    order: RunOrder = RunOrder.CATALOG_FIRST,
    dry_run: bool = False,
    progress: bool = True,
    progress_every: int | None = None,
    tracking_uri: str | None = None,
    approve_large_run: bool = False,
    override: bool = False,
) -> None:
    """Execute F2 suite experiments."""
    from repro_core.cli.commands import run_command

    suite_def = DEFINITION.get_suite(suite)
    requires_approval = getattr(suite_def, "large_run_requires_approval", True)
    if requires_approval and not dry_run and not approve_large_run:
        raise ValueError(f"canonical {suite} training requires --approve-large-run")
    resolved_uri = resolve_tracking_uri(tracking_uri) if not dry_run else None

    def run_fn(*args, **kwargs):
        return run_tracked_yaml(*args, override=override, **kwargs)

    run_command(
        suite_def,
        experiments=experiment or [],
        all_experiments=all_experiments,
        atomic_runs=atomic_run or [],
        excluded_atomic_runs=exclude_atomic_run or [],
        seed_set=seed_set,
        seeds=seed,
        device=device,
        override_values=override_values or [],
        order=order,
        dry_run=dry_run,
        progress=progress,
        progress_every=progress_every,
        tracking_uri=resolved_uri,
        run_fn=run_fn,
    )


__all__ = ["run_command_impl"]
