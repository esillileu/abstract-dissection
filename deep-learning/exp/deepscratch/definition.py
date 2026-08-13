"""DeepScratch domain definition and volume dispatch."""

from __future__ import annotations

from dataclasses import dataclass

from exp.domain import DomainDefinition
from exp.ds1.cli import DEFINITION as DS1_IMPLEMENTED
from exp.ds1_original.cli import DEFINITION as DS1_ORIGINAL
from exp.ds2.cli import DEFINITION as DS2_IMPLEMENTED
from exp.ds2_original.cli import DEFINITION as DS2_ORIGINAL

from .identity import Variant, Volume


@dataclass(frozen=True)
class DeepScratchDefinition:
    name: str = "deepscratch"

    def implementation(self, volume: Volume, variant: Variant) -> DomainDefinition:
        return {
            (Volume.DS1, Variant.IMPLEMENTED): DS1_IMPLEMENTED,
            (Volume.DS1, Variant.ORIGINAL): DS1_ORIGINAL,
            (Volume.DS2, Variant.IMPLEMENTED): DS2_IMPLEMENTED,
            (Volume.DS2, Variant.ORIGINAL): DS2_ORIGINAL,
        }[(volume, variant)]


DEFINITION = DeepScratchDefinition()
