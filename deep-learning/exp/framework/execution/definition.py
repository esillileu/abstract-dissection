"""Generic execution identities and plan contracts."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RunOrder(str, Enum):
    CATALOG_FIRST = "catalog-first"
    SEED_FIRST = "seed-first"


@dataclass(frozen=True)
class ExecutionDefinition:
    name: str
    config_root: Path
    spec_module: str
    executor_module: str
    checkpoint_source_resolver_module: str | None = None
    default_seed_set: str = "research_v1"
    domain: str | None = None
    suite: str | None = None
    variant: str | None = None

    def load_run_spec(
        self,
        path: Path,
        *,
        atomic_run_id: str,
        overrides: dict[str, object],
    ) -> Any:
        return importlib.import_module(self.spec_module).parse_run_spec(
            path, atomic_run_id=atomic_run_id, overrides=overrides
        )


@dataclass(frozen=True)
class RunSelection:
    experiment_ids: tuple[str, ...] = ()
    all_experiments: bool = False
    atomic_run_ids: tuple[str, ...] = ()
    excluded_atomic_run_ids: tuple[str, ...] = ()
    seed_values: str | None = None
    seed_set: str | None = None


@dataclass(frozen=True)
class RunOptions:
    device: str | None = None
    overrides: dict[str, object] = field(default_factory=dict)
    progress: str = "auto"
    progress_every: int = 10
    order: RunOrder = RunOrder.CATALOG_FIRST


@dataclass(frozen=True)
class RunPlan:
    domain: str
    experiment_id: str
    path: Path
    atomic_run_id: str
    seed: int | None
    device: str
