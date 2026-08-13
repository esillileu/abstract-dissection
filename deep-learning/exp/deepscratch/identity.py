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


LEGACY_NAMESPACES = {
    (Volume.DS1, Variant.IMPLEMENTED): "ds1",
    (Volume.DS1, Variant.ORIGINAL): "ds1_original",
    (Volume.DS2, Variant.IMPLEMENTED): "ds2",
    (Volume.DS2, Variant.ORIGINAL): "ds2_original",
}


@dataclass(frozen=True)
class DeepScratchCoordinate:
    volume: Volume
    experiment_id: str
    condition_id: str
    variant: Variant

    @property
    def atomic_run_id(self) -> str:
        """Compatibility alias; the canonical concept is condition."""
        return self.condition_id

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


def legacy_namespace(volume: Volume, variant: Variant) -> str:
    return LEGACY_NAMESPACES[(volume, variant)]
