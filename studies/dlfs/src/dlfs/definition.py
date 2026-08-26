"""DeepScratch domain definition and volume dispatch."""

from __future__ import annotations

from dataclasses import dataclass

from repro_core.execution import ExecutionDefinition
from repro_core.registry import CommandGroups

from .ds1.catalog import IMPLEMENTED as DS1_IMPLEMENTED
from .ds1.catalog import ORIGINAL as DS1_ORIGINAL
from .ds2.catalog import IMPLEMENTED as DS2_IMPLEMENTED
from .ds2.catalog import ORIGINAL as DS2_ORIGINAL
from .identity import Variant, Volume


@dataclass(frozen=True)
class DeepScratchDefinition:
    name: str = "deepscratch"

    def implementation(self, volume: Volume, variant: Variant) -> ExecutionDefinition:
        return {
            (Volume.DS1, Variant.IMPLEMENTED): DS1_IMPLEMENTED,
            (Volume.DS1, Variant.ORIGINAL): DS1_ORIGINAL,
            (Volume.DS2, Variant.IMPLEMENTED): DS2_IMPLEMENTED,
            (Volume.DS2, Variant.ORIGINAL): DS2_ORIGINAL,
        }[(volume, variant)]

    def register_commands(self, groups: CommandGroups) -> None:
        """Attach all domain-owned commands to the generic composition root."""
        from . import cli

        groups.plan.command(self.name)(cli.plan)
        groups.run.command(self.name)(cli.run)
        groups.analyze.command(self.name)(cli.analyze)
        groups.check.command(self.name)(cli.check)
        groups.profile.command(self.name)(cli.profile)


DEFINITION = DeepScratchDefinition()
