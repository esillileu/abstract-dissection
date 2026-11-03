"""Storage layer for local run management in F2.

Implements ref.md section 4.1 & 4.2:
- Local run layout under workspace/runs/f2/<suite>/<slot_id>/<attempt_id>/
- Attempt discovery, status checking, and recovery targeting
- Preserves local execution results on override/rerun
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from f2.execution.identity import ConditionIdentity
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
    config_digest: str | None = None
    seed: int | None = None
    slot_id: str | None = None


class LocalRunStore:
    """Manages local attempt paths and state queries for F2 suites."""

    def __init__(self, paths: RuntimePaths | None = None) -> None:
        self._paths = paths or RuntimePaths.from_environment()
        self._locator = PathRunLocator()

    def discover_attempts(self, suite_id: str, slot_id: str) -> list[Path]:
        """Discover existing attempts for a given slot, sorted chronologically."""
        base_dir = self._paths.runs("f2", suite_id, slot_id)
        if not base_dir.exists():
            return []

        # Find attempt subdirectories matching run_XX or attempt_*
        subdirs: list[tuple[int, Path]] = []
        for p in base_dir.iterdir():
            if not p.is_dir():
                continue
            match = re.match(r"^run_(\d+)$", p.name)
            if match:
                subdirs.append((int(match.group(1)), p))
            elif p.name.startswith("attempt_"):
                # fallback for named attempt folders
                subdirs.append((9999, p))

        if subdirs:
            subdirs.sort(key=lambda item: (item[0], item[1].name))
            return [p for _, p in subdirs]

        # If base_dir itself has run files directly (legacy/initial layout), treat base_dir as attempt
        if any(base_dir.iterdir()):
            return [base_dir]
        return []

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
        config_file = run_dir / "config.json"

        config_digest: str | None = None
        seed: int | None = None
        slot_id: str | None = None

        # Try to read identity from result.json or config.json
        data_src = None
        if result_file.is_file():
            try:
                data_src = json.loads(result_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        if data_src is None and config_file.is_file():
            try:
                data_src = json.loads(config_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        if isinstance(data_src, dict):
            ident = data_src.get("identity")
            if isinstance(ident, dict):
                config_digest = ident.get("config_digest")
                seed_val = ident.get("seed")
                if seed_val is not None:
                    try:
                        seed = int(seed_val)
                    except ValueError:
                        pass
                slot_id = ident.get("planned_run_slot_id")

        return LocalRunStatus(
            run_dir=run_dir,
            location=loc,
            is_complete=loc.is_final,
            is_checkpoint=loc.is_checkpoint,
            checkpoint_path=loc.checkpoint_path,
            epoch=loc.epoch,
            metrics_path=metrics_file if metrics_file.exists() else None,
            result_path=result_file if result_file.exists() else None,
            config_digest=config_digest,
            seed=seed,
            slot_id=slot_id,
        )

    def matches_condition(
        self,
        status: LocalRunStatus,
        condition: ConditionIdentity,
    ) -> bool:
        """Check if local status matches the given condition identity."""
        if not status.run_dir.exists():
            return False
        # If config_digest or seed is present in local status, they must match condition
        if (
            status.config_digest is not None
            and status.config_digest != condition.config_digest
        ):
            return False
        if status.seed is not None and status.seed != condition.seed:
            return False
        if (
            status.slot_id is not None
            and condition.planned_run_slot_id
            and status.slot_id != condition.planned_run_slot_id
        ):
            return False
        # If neither result nor config has identity information recorded, verify if config.json exists
        config_file = status.run_dir / "config.json"
        result_file = status.run_dir / "result.json"
        if not config_file.is_file() and not result_file.is_file():
            return False
        return True

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
