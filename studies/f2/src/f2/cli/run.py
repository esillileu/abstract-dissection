"""Run command implementation executing F2 experiment runs via Coordinator."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from repro_core.context import RuntimePaths
from repro_core.execution.definition import RunOrder

from ..execution.definitions import DEFINITION
from ..tracking import resolve_tracking_uri


def execute_f2_plan(
    path: str | Path,
    *,
    atomic_run_id: str | None = None,
    seed: int | None = None,
    device: str | None = None,
    overrides: dict[str, object] | None = None,
    executor_module: str | None = None,
    spec_module: str | None = None,
    tracking_uri: str,
    progress_reporter: Any | None = None,
    override: bool = False,
    **_: object,
) -> Any:
    """Execute one planned run using F2 execution spec and suite executor directly."""
    if spec_module is None or executor_module is None:
        raise ValueError("F2 execution requires spec and executor modules")

    parser = importlib.import_module(spec_module)
    spec = parser.parse_run_spec(path, atomic_run_id=atomic_run_id, overrides=overrides)
    spec = spec.with_seed(int(spec.identity["seed"]) if seed is None else seed)
    config = spec.to_executor_config()

    if device not in {None, "cpu"}:
        raise ValueError("canonical W2V parity execution requires CPU")

    paths = RuntimePaths.from_environment()
    is_override = (
        override
        or bool(config.get("override", False))
        or bool(overrides.get("override", False) if overrides else False)
    )

    exec_mod = importlib.import_module(executor_module)
    kind = str(config.get("kind", "word2vec"))
    resolver = getattr(exec_mod, "get_executor", None)
    if callable(resolver):
        executor = resolver(kind)
    else:
        executors = getattr(exec_mod, "EXECUTORS", {})
        executor = executors[kind]

    return executor.execute_spec(
        config,
        tracking_uri=tracking_uri,
        paths=paths,
        override=is_override,
        progress_reporter=progress_reporter,
    )


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

    def run_fn(*args: Any, **kwargs: Any) -> Any:
        return execute_f2_plan(*args, override=override, **kwargs)

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


__all__ = ["execute_f2_plan", "run_command_impl"]
