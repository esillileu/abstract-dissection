"""F2 study domain definition, volume execution definitions, and command dispatcher."""

from __future__ import annotations

from dataclasses import dataclass

from repro_core.registry import CommandGroups


@dataclass(frozen=True)
class F2Definition:
    name: str = "f2"

    def register_commands(self, groups: CommandGroups) -> None:
        """Attach domain-owned commands to repro CLI."""
        from .catalog import cli as catalog_cli
        from .corpus import cli as corpus_cli

        groups.register_study(self.name, corpus_cli.app)
        groups.register_study(self.name, catalog_cli.app)


DEFINITION = F2Definition()

__all__ = ["DEFINITION", "F2Definition"]
