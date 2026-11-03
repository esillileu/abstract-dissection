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
from typing import TYPE_CHECKING, Any

from repro_core.context.paths import RuntimePaths
from repro_core.execution.locator import PathRunLocator, RunLocation

if TYPE_CHECKING:
    from f2.execution.identity import ConditionIdentity


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

        # Check if base_dir itself has run files directly (e.g. initial run before override created subdirs)
        base_has_direct_run_files = (base_dir / "result.json").is_file() or (
            base_dir / "checkpoints"
        ).is_dir()

        attempts: list[Path] = []
        if base_has_direct_run_files:
            attempts.append(base_dir)

        if subdirs:
            subdirs.sort(key=lambda item: (item[0], item[1].name))
            attempts.extend([p for _, p in subdirs])

        if attempts:
            return attempts

        # If base_dir has any files at all, return base_dir
        if any(base_dir.iterdir()):
            return [base_dir]
        return []

    def allocate_new_attempt(
        self,
        suite_id: str,
        slot_id: str,
    ) -> Path:
        """Atomically allocate a fresh, non-existent indexed attempt directory (run_XX)."""
        base_dir = self._paths.runs("f2", suite_id, slot_id)
        base_dir.mkdir(parents=True, exist_ok=True)
        idx = 1
        while True:
            candidate = base_dir / f"run_{idx:02d}"
            try:
                candidate.mkdir(parents=False, exist_ok=False)
                return candidate
            except FileExistsError:
                idx += 1

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
            return self.allocate_new_attempt(suite_id, slot_id)

        return base_dir

    def get_isolated_run_root(
        self,
        suite_id: str,
        slot_id: str,
    ) -> Path:
        """Resolve a new, isolated attempt directory when an existing directory belongs to a conflicting condition.

        Prevents overwriting or reusing existing runs when config_digest/identity differs.
        """
        base_dir = self._paths.runs("f2", suite_id, slot_id)
        if not base_dir.exists() or not any(base_dir.iterdir()):
            return base_dir

        return self.allocate_new_attempt(suite_id, slot_id)

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

        res_data = None
        if result_file.is_file():
            try:
                res_data = json.loads(result_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        cfg_data = None
        if config_file.is_file():
            try:
                cfg_data = json.loads(config_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Check for conflicts between result.json and config.json if both exist
        ident_res = res_data.get("identity") if isinstance(res_data, dict) else None
        ident_cfg = cfg_data.get("identity") if isinstance(cfg_data, dict) else None

        if isinstance(ident_res, dict) and isinstance(ident_cfg, dict):
            # If both exist and conflict on essential keys, mark as invalid/unmatched
            for key in ("config_digest", "seed", "planned_run_slot_id"):
                v_res = ident_res.get(key)
                v_cfg = ident_cfg.get(key)
                if v_res is not None and v_cfg is not None and str(v_res) != str(v_cfg):
                    return LocalRunStatus(
                        run_dir=run_dir,
                        location=loc,
                        is_complete=False,
                        is_checkpoint=False,
                        checkpoint_path=None,
                        epoch=None,
                        metrics_path=None,
                        result_path=None,
                        config_digest=None,
                        seed=None,
                        slot_id=None,
                    )

        target_ident = ident_res or ident_cfg or {}
        config_digest = target_ident.get("config_digest")
        seed = None
        seed_val = target_ident.get("seed")
        if seed_val is not None:
            try:
                seed = int(seed_val)
            except ValueError:
                pass
        slot_id = target_ident.get("planned_run_slot_id")

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
        """Check if local status strictly matches the given condition identity."""
        if not status.run_dir.exists():
            return False

        # Strict identity requirement: essential fields MUST be present and match
        if not status.config_digest or status.config_digest != condition.config_digest:
            return False
        if status.seed is None or status.seed != condition.seed:
            return False
        if condition.planned_run_slot_id and (
            not status.slot_id or status.slot_id != condition.planned_run_slot_id
        ):
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
