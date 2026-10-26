"""Local file storage, checkpoint persistence, and retention for run outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from repro_core.context.checkpoint import CheckpointRetentionPolicy


@dataclass
class LocalStore:
    """Manages file storage, checkpoints, and retention in a dedicated run directory."""

    run_root: Path
    retention_policy: CheckpointRetentionPolicy = field(
        default_factory=CheckpointRetentionPolicy
    )

    def __post_init__(self) -> None:
        self.run_root = Path(self.run_root)

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_root / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        return self.run_root / "metrics"

    @property
    def lookup_dir(self) -> Path:
        return self.run_root / "lookup"

    def ensure_directories(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def write_atomic_json(self, destination: Path, data: Any) -> Path:
        """Atomically write JSON data to destination."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(f"{destination.suffix}.tmp")
        tmp.write_text(
            json.dumps(data, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        tmp.replace(destination)
        return destination

    def write_result_report(self, report: dict[str, Any]) -> Path:
        return self.write_atomic_json(self.run_root / "result.json", report)

    def load_result_report(self) -> dict[str, Any]:
        result_path = self.run_root / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"result.json not found in {self.run_root}")
        return json.loads(result_path.read_text(encoding="utf-8"))

    def write_config(self, config: dict[str, Any]) -> Path:
        return self.write_atomic_json(self.run_root / "config.json", config)

    def write_epoch_reports(self, reports: list[dict[str, Any]]) -> Path:
        return self.write_atomic_json(self.metrics_dir / "epoch_reports.json", reports)


__all__ = ["LocalStore"]
