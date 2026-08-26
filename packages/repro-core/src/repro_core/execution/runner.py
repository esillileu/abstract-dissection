"""Preflight checks and sequential execution for planned runs."""

from __future__ import annotations

import importlib
import os
import shutil
import warnings
from collections.abc import Callable
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from .definition import ExecutionDefinition, RunOptions, RunPlan


class Runner:
    def __init__(
        self,
        domain: ExecutionDefinition,
        run_fn: Callable[..., object] | None = None,
    ) -> None:
        self.domain = domain
        self._run_fn = run_fn

    def run(
        self,
        plans: list[RunPlan],
        options: RunOptions,
        *,
        run_fn: Callable[..., object] | None = None,
    ) -> None:
        runner_fn = run_fn or self._run_fn
        if runner_fn is None:
            raise ValueError(
                "Runner requires an execution runner function (e.g. run_yaml)"
            )
        self._require_tracking_server(plans, options.overrides)
        self._require_devices(plans)
        from repro_core.context.progress import ProgressManager, RunProgressContext

        progress = ProgressManager(
            mode=options.progress, every=options.progress_every, total_runs=len(plans)
        )
        receipts = []
        checkpoint_source_resolver = None
        if self.domain.checkpoint_source_resolver_module is not None:
            checkpoint_source_resolver = importlib.import_module(
                self.domain.checkpoint_source_resolver_module
            ).resolve_checkpoint_source
        try:
            for index, plan in enumerate(plans, start=1):
                spec = self.domain.load_run_spec(
                    plan.path,
                    atomic_run_id=plan.atomic_run_id,
                    overrides=options.overrides,
                )
                config = spec.to_executor_config()
                training = config.get("training", {})
                total_updates = None
                if (
                    isinstance(training, dict)
                    and training.get("max_updates") is not None
                ):
                    total_updates = int(training["max_updates"])
                context = RunProgressContext(
                    label=f"{plan.experiment_id}/{plan.atomic_run_id}/s{'single' if plan.seed is None else plan.seed}",
                    index=index,
                    count=len(plans),
                    total_updates=total_updates,
                )
                reporter = progress.reporter(context)
                progress.on_run_start(context)
                try:
                    run_kwargs = {}
                    if checkpoint_source_resolver is not None:
                        run_kwargs["checkpoint_source_resolver"] = (
                            checkpoint_source_resolver
                        )
                    receipt = runner_fn(
                        plan.path,
                        atomic_run_id=plan.atomic_run_id,
                        seed=plan.seed,
                        device=plan.device,
                        overrides=options.overrides,
                        executor_module=self.domain.executor_module,
                        spec_module=self.domain.spec_module,
                        progress_reporter=reporter,
                        **run_kwargs,
                    )
                    receipts.append(receipt)
                finally:
                    reporter.close()
                    progress.on_run_end()
        finally:
            progress.close()
            for receipt in receipts:
                if not getattr(receipt, "durable_complete", False):
                    continue
                staging_root = getattr(receipt, "staging_root", None)
                if staging_root is not None:
                    _remove_durable_staging(staging_root)

    def _require_tracking_server(
        self, plans: list[RunPlan], overrides: dict[str, object]
    ) -> None:
        uris = set()
        for plan in plans:
            config = self.domain.load_run_spec(
                plan.path, atomic_run_id=plan.atomic_run_id, overrides=overrides
            ).to_executor_config()
            tracking = config.get("tracking", {})
            if isinstance(tracking, dict) and tracking.get("enabled", True):
                uris.add(
                    os.getenv("REPRO_TRACKING_URI")
                    or os.getenv("MLFLOW_TRACKING_URI")
                    or str(tracking.get("uri", "http://127.0.0.1:5000"))
                )
        for uri in sorted(uris):
            try:
                with urlopen(f"{uri.rstrip('/')}/health", timeout=5) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"Tracking server health check returned HTTP {response.status}"
                        )
            except (OSError, URLError) as exc:
                raise RuntimeError(
                    f"Tracking server is unavailable at {uri}. Start it before running plans."
                ) from exc

    @staticmethod
    def _require_devices(plans: list[RunPlan]) -> None:
        from repro_core.numerics import BackendConfig, make_backend

        for device in sorted({plan.device for plan in plans}):
            try:
                make_backend(BackendConfig(device=device, dtype="float32", seed=0))
            except Exception as exc:
                raise RuntimeError(
                    f"requested device is unavailable: {device}"
                ) from exc


def _remove_durable_staging(staging_root: Path) -> None:
    """Remove verified staging, tolerating an already-completed cleanup."""
    try:
        shutil.rmtree(staging_root)
    except FileNotFoundError:
        # Cleanup is intentionally idempotent. A previous run or another
        # cleanup worker may already have removed it.
        return
    except OSError as exc:
        warnings.warn(
            f"verified staging was preserved after cleanup failure: "
            f"{staging_root}: {exc}",
            stacklevel=2,
        )


def print_plans(plans: list[RunPlan]) -> None:
    print(f"{plans[0].domain}: {len(plans)} planned runs")
    for plan in plans:
        seed = "single" if plan.seed is None else plan.seed
        print(
            f"{plan.experiment_id} {plan.path.name} {plan.atomic_run_id} seed={seed} device={plan.device}"
        )


def run_config(
    config: dict[str, object],
    context: object | None = None,
    *,
    executor_module: str | None = None,
) -> object:
    """Run through an explicitly selected experiment-domain adapter."""
    from repro_core.context import ExperimentContext

    if executor_module is None:
        raise ValueError("run_config requires an executor_module")
    module = importlib.import_module(executor_module)
    ctx = (
        context
        if isinstance(context, ExperimentContext)
        else (ExperimentContext() if context is None else context)
    )
    kind = str(config["kind"])
    if kind == "observation":
        resolver = getattr(module, "get_observation_executor", None)
        if callable(resolver):
            return resolver(config).run(config, ctx)
        raise ValueError(
            f"executor module '{executor_module}' does not support kind 'observation'"
        )

    resolver = getattr(module, "get_executor", None)
    if callable(resolver):
        return resolver(kind).run(config, ctx)

    executors = getattr(module, "EXECUTORS", None)
    if isinstance(executors, dict) and kind in executors:
        executor = executors[kind]
        return (executor() if isinstance(executor, type) else executor).run(config, ctx)

    raise ValueError(
        f"unknown experiment kind '{kind}' in executor module '{executor_module}'"
    )
