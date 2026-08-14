"""Preflight checks and sequential execution for planned runs."""

from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from .definition import ExecutionDefinition, RunOptions, RunPlan


class Runner:
    def __init__(self, domain: ExecutionDefinition) -> None:
        self.domain = domain

    def run(self, plans: list[RunPlan], options: RunOptions) -> None:
        self._require_mlflow_server(plans, options.overrides)
        self._require_devices(plans)
        from mlprosection.experiment.progress import ProgressManager, RunProgressContext
        from mlprosection_mlflow import run_yaml

        progress = ProgressManager(mode=options.progress, every=options.progress_every, total_runs=len(plans))
        receipts = []
        try:
            for index, plan in enumerate(plans, start=1):
                spec = self.domain.load_run_spec(plan.path, atomic_run_id=plan.atomic_run_id, overrides=options.overrides)
                config = spec.to_executor_config()
                training = config.get("training", {})
                total_updates = None
                if isinstance(training, dict) and training.get("max_updates") is not None:
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
                    receipt = run_yaml(
                        plan.path,
                        atomic_run_id=plan.atomic_run_id,
                        seed=plan.seed,
                        device=plan.device,
                        overrides=options.overrides,
                        executor_module=self.domain.executor_module,
                        spec_module=self.domain.spec_module,
                        progress_reporter=reporter,
                    )
                    receipts.append(receipt)
                finally:
                    reporter.close()
                    progress.on_run_end()
        finally:
            progress.close()
            for receipt in receipts:
                if not receipt.durable_complete:
                    continue
                _remove_durable_staging(receipt.staging_root)

    def _require_mlflow_server(self, plans: list[RunPlan], overrides: dict[str, object]) -> None:
        uris = set()
        for plan in plans:
            config = self.domain.load_run_spec(plan.path, atomic_run_id=plan.atomic_run_id, overrides=overrides).to_executor_config()
            tracking = config.get("tracking", {})
            if isinstance(tracking, dict) and tracking.get("enabled", True):
                uris.add(os.getenv("MLFLOW_TRACKING_URI") or str(tracking.get("uri", "http://127.0.0.1:5000")))
        for uri in sorted(uris):
            try:
                with urlopen(f"{uri.rstrip('/')}/health", timeout=5) as response:
                    if response.status != 200:
                        raise RuntimeError(f"MLflow health check returned HTTP {response.status}")
            except (OSError, URLError) as exc:
                raise RuntimeError(f"MLflow server is unavailable at {uri}. Start it before running plans.") from exc

    @staticmethod
    def _require_devices(plans: list[RunPlan]) -> None:
        from mlprosection.core.backend import BackendConfig, make_backend

        for device in sorted({plan.device for plan in plans}):
            try:
                make_backend(BackendConfig(device=device, dtype="float32", seed=0))
            except Exception as exc:
                raise RuntimeError(f"requested device is unavailable: {device}") from exc


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
        print(f"{plan.experiment_id} {plan.path.name} {plan.atomic_run_id} seed={seed} device={plan.device}")
