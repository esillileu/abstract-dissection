"""Typed ownership and workspace path resolution.

Only this module reads experiment cache/artifact root environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class StateOwner(str, Enum):
    CACHE = "cache"
    ARTIFACT = "artifact"


@dataclass(frozen=True)
class StateCoordinate:
    domain: str
    volume: str
    experiment: str
    variant: str
    purpose: str | None = None

    def parts(self) -> tuple[str, ...]:
        values = (self.domain, self.volume, self.experiment, self.variant)
        _validate_parts(values)
        if self.purpose is None:
            return values
        _validate_parts((self.purpose,))
        return (*values, self.purpose)


@dataclass(frozen=True)
class WorkspacePaths:
    cache_root: Path
    artifact_root: Path

    @classmethod
    def from_environment(cls, repository_root: Path) -> WorkspacePaths:
        repository_root = repository_root.resolve()
        return cls(
            Path(os.getenv("EXP_CACHE_ROOT", repository_root / ".cache/experiments")),
            Path(os.getenv("EXP_ARTIFACT_ROOT", repository_root / ".artifacts/experiments")),
        )

    def resolve(self, owner: StateOwner, coordinate: StateCoordinate) -> Path:
        root = self.cache_root if owner is StateOwner.CACHE else self.artifact_root
        return root.joinpath(*coordinate.parts())


def _validate_parts(parts: tuple[str, ...]) -> None:
    for part in parts:
        if not part or part in {".", ".."} or Path(part).name != part:
            raise ValueError(f"invalid state coordinate component: {part!r}")
