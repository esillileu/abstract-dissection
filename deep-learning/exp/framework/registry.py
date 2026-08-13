"""Small domain-neutral plugin registry."""

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


class DomainPlugin(Protocol):
    """A domain owns its vocabulary and command dispatch."""

    name: str

    def register_commands(self, groups: CommandGroups) -> None: ...


@dataclass
class DomainRegistry:
    _plugins: dict[str, DomainPlugin]

    def __init__(self) -> None:
        self._plugins = {}

    def register(
        self,
        plugin: DomainPlugin,
        groups: CommandGroups | None = None,
    ) -> None:
        if plugin.name in self._plugins:
            raise ValueError(f"duplicate domain plugin: {plugin.name}")
        self._plugins[plugin.name] = plugin
        if groups is not None:
            plugin.register_commands(groups)

    def get(self, name: str) -> DomainPlugin:
        try:
            return self._plugins[name]
        except KeyError as exc:
            raise ValueError(f"unknown domain plugin: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._plugins)
