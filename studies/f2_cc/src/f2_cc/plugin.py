"""F2 Common Crawl study plugin."""

from __future__ import annotations

from dataclasses import dataclass

from repro_core.registry import CommandGroups


@dataclass(frozen=True)
class F2CCStudyPlugin:
    name: str = "f2-cc"
    display_name: str = "F2 Common Crawl Corpus Producer"
    description: str = "Acquire, audit, validate, and publish Common Crawl releases"

    def register_commands(self, groups: CommandGroups) -> None:
        from .cli import app

        groups.root.add_typer(app, name=self.name)


PLUGIN = F2CCStudyPlugin()
