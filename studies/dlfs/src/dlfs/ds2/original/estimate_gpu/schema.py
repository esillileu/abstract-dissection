"""Configuration paths and data schemas for DS2 GPU runtime estimation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from repro_core.context.paths import StateCoordinate, StateOwner, WorkspacePaths

BOOK_ROOT = Path("01_deep-learning-from-base/deep-learning-from-scratch-2").resolve()
B2_SOURCE_ROOT = Path("01_deep-learning-from-base/src").resolve()
PTB_TRAIN = B2_SOURCE_ROOT / "datasets/ptb.train.npy"
DEFAULT_OUTPUT = (
    WorkspacePaths.from_environment(Path.cwd()).resolve(
        StateOwner.CACHE,
        StateCoordinate("deepscratch", "ds2", "runtime-estimate", "original", "gpu"),
    )
    / "estimate.json"
)


@dataclass(frozen=True)
class RuntimeEstimate:
    experiment_id: str
    condition: str
    source: str
    benchmark_units: int
    benchmark_unit: str
    seconds_per_unit: float
    total_units: int
    projected_compute_time_s: float
    projected_overhead_time_s: float
    projected_total_time_s: float


def _add_import_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
