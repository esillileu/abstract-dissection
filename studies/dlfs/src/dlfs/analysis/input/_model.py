"""Analysis run data structures for DeepScratch study renderers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from repro_core.results import NativeRunResult

from ...identity import Variant


@dataclass(frozen=True)
class AnalysisRun:
    run_id: str
    canonical_condition_id: str
    native_condition_id: str
    seed: str
    variant: Variant
    result: NativeRunResult
    local_artifact_root: Path | None = None
