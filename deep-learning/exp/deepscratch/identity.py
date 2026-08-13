"""Canonical DeepScratch logical identity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Volume(str, Enum):
    DS1 = "ds1"
    DS2 = "ds2"


class Variant(str, Enum):
    IMPLEMENTED = "implemented"
    ORIGINAL = "original"


@dataclass(frozen=True)
class DeepScratchCoordinate:
    volume: Volume
    experiment_id: str
    condition_id: str
    variant: Variant

    @property
    def mlflow_experiment(self) -> str:
        return f"deepscratch.{self.volume.value}"

    def metadata(
        self,
        *,
        schema_name: str,
        schema_version: int,
        protocol_version: str,
        comparison_condition_id: str | None = None,
    ) -> dict[str, str]:
        metadata = {
            "domain.name": "deepscratch",
            "deepscratch.volume": self.volume.value,
            "implementation.variant": self.variant.value,
            "experiment.id": self.experiment_id,
            "condition.id": self.condition_id,
            "atomic_run.id": self.condition_id,
            "result.schema.name": schema_name,
            "result.schema.version": str(schema_version),
            "protocol.version": protocol_version,
        }
        if comparison_condition_id is not None:
            metadata["comparison.condition_id"] = comparison_condition_id
        return metadata
