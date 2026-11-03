"""Storage layer for local run management in F2.

Implements ref.md section 4.1 & 4.2:
- Local run layout under workspace/runs/f2/<suite>/<slot_id>/<attempt_id>/
- Attempt discovery, status checking, and recovery targeting
- Preserves local execution results on override/rerun
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repro_core.context.paths import RuntimePaths
from repro_core.execution.locator import PathRunLocator, RunLocation


@dataclass(frozen=True)
class LocalRunStatus:
    """Status of a local run attempt."""

    run_dir: Path
    location: RunLocation
    is_complete: bool
    is_checkpoint: bool
    checkpoint_path: Path | None
    epoch: int | None
    metrics_path: Path | None
    result_path: Path | None


class LocalRunStore:
    """Manages local attempt paths and state queries for F2 suites."""

    def __init__(self, paths: RuntimePaths | None = None) -> None:
        self._paths = paths or RuntimePaths.from_environment()
        self._locator = PathRunLocator()

    def get_run_root(
        self,
        suite_id: str,
        slot_id: str,
        *,
        attempt_id: str | None = None,
        override: bool = False,
    ) -> Path:
        """Resolve directory path for a slot attempt."""
        base_dir = self._paths.runs("f2", suite_id, slot_id)
        if attempt_id:
            return base_dir / attempt_id

        if not override:
            return base_dir

        # For override, create or resolve a new indexed attempt directory if base exists
        if base_dir.exists() and any(base_dir.iterdir()):
            idx = 1
            while (base_dir / f"run_{idx:02d}").exists() and any(
                (base_dir / f"run_{idx:02d}").iterdir()
            ):
                idx += 1
            return base_dir / f"run_{idx:02d}"

        return base_dir

    def inspect_run(self, run_dir: Path) -> LocalRunStatus:
        """Inspect the current state of a local run directory."""
        if not run_dir.exists():
            loc = self._locator.locate(run_dir)
            return LocalRunStatus(
                run_dir=run_dir,
                location=loc,
                is_complete=False,
                is_checkpoint=False,
                checkpoint_path=None,
                epoch=None,
                metrics_path=None,
                result_path=None,
            )

        loc = self._locator.locate(run_dir)
        metrics_file = run_dir / "metrics" / "observations.csv"
        if not metrics_file.exists():
            metrics_file = run_dir / "metrics" / "metrics.json"
        if not metrics_file.exists():
            metrics_file = run_dir / "metrics.json"

        result_file = run_dir / "result.json"

        return LocalRunStatus(
            run_dir=run_dir,
            location=loc,
            is_complete=loc.is_final,
            is_checkpoint=loc.is_checkpoint,
            checkpoint_path=loc.checkpoint_path,
            epoch=loc.epoch,
            metrics_path=metrics_file if metrics_file.exists() else None,
            result_path=result_file if result_file.exists() else None,
        )

    def load_result_summary(self, run_dir: Path) -> dict[str, Any] | None:
        """Load result.json if present."""
        res_file = run_dir / "result.json"
        if not res_file.exists():
            return None
        try:
            with open(res_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None


__all__ = ["LocalRunStatus", "LocalRunStore"]
