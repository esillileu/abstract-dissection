"""Small domain-neutral plugin registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class DomainPlugin(Protocol):
    """A domain owns its vocabulary and command dispatch."""

    name: str


@dataclass
class DomainRegistry:
    _plugins: dict[str, DomainPlugin]

    def __init__(self) -> None:
        self._plugins = {}

    def register(self, plugin: DomainPlugin) -> None:
        if plugin.name in self._plugins:
            raise ValueError(f"duplicate domain plugin: {plugin.name}")
        self._plugins[plugin.name] = plugin

    def get(self, name: str) -> DomainPlugin:
        try:
            return self._plugins[name]
        except KeyError as exc:
            raise ValueError(f"unknown domain plugin: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._plugins)
