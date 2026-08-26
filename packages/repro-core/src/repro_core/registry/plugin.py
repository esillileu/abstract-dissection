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


_EXECUTORS: dict[str, Any] = {}


def register_executor(kind: str):
    def decorator(executor: Any) -> Any:
        if kind in _EXECUTORS:
            raise ValueError(f"executor already registered: {kind}")
        _EXECUTORS[kind] = executor() if isinstance(executor, type) else executor
        return executor

    return decorator


def get_executor(kind: str) -> Any:
    try:
        return _EXECUTORS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown experiment kind: {kind}") from exc


__all__ = [
    "CommandGroups",
    "DomainPlugin",
    "DomainRegistry",
    "StudyPlugin",
    "get_executor",
    "register_executor",
]
