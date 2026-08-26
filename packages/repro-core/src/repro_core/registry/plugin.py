"""Study-neutral plugin registry and command group bindings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class CommandGroups:
    root: Any
    plan: Any
    run: Any
    analyze: Any
    check: Any
    profile: Any
    storage: Any


class StudyPlugin(Protocol):
    """A study owns its vocabulary, experiments, and command dispatch."""

    name: str

    def register_commands(self, groups: CommandGroups) -> None: ...


DomainPlugin = StudyPlugin


@dataclass
class DomainRegistry:
    _plugins: dict[str, StudyPlugin]

    def __init__(self) -> None:
        self._plugins = {}

    def register(
        self,
        plugin: StudyPlugin,
        groups: CommandGroups | None = None,
    ) -> None:
        if plugin.name in self._plugins:
            raise ValueError(f"duplicate domain plugin: {plugin.name}")
        self._plugins[plugin.name] = plugin
        if groups is not None:
            plugin.register_commands(groups)

    def get(self, name: str) -> StudyPlugin:
        try:
            return self._plugins[name]
        except KeyError as exc:
            raise ValueError(f"unknown domain plugin: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._plugins)


__all__ = [
    "CommandGroups",
    "DomainPlugin",
    "DomainRegistry",
    "StudyPlugin",
]
