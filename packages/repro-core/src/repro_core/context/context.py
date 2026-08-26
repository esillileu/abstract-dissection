"""Execution context and runtime environment bindings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import RuntimePaths


@dataclass
class ExperimentContext:
    metadata: dict[str, Any] = field(default_factory=dict)
    paths: RuntimePaths = field(default_factory=RuntimePaths.from_environment)

    @property
    def artifact_root(self) -> Path | None:
        val = self.metadata.get("artifact_root")
        return Path(val) if val else None

    @property
    def checkpoint_root(self) -> Path | None:
        val = self.metadata.get("checkpoint_root")
        return Path(val) if val else None


RunContext = ExperimentContext
