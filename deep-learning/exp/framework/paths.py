"""Typed ownership and workspace path resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class StateOwner(str, Enum):
    RESULT = "result"
    CACHE = "cache"
    ARTIFACT = "artifact"
    LEGACY = "legacy"


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
    result_staging_root: Path
    cache_root: Path
    artifact_root: Path
    legacy_root: Path

    @classmethod
    def from_environment(cls, repository_root: Path) -> WorkspacePaths:
        repository_root = repository_root.resolve()
        return cls(
            Path(os.getenv("EXP_RESULT_STAGING_ROOT", repository_root / "results/experiments")),
            Path(os.getenv("EXP_CACHE_ROOT", repository_root / ".cache/experiments")),
            Path(os.getenv("EXP_ARTIFACT_ROOT", repository_root / ".artifacts/experiments")),
            Path(os.getenv("EXP_LEGACY_ROOT", repository_root / ".legacy/experiments")),
        )

    def resolve(self, owner: StateOwner, coordinate: StateCoordinate) -> Path:
        root = {
            StateOwner.RESULT: self.result_staging_root,
            StateOwner.CACHE: self.cache_root,
            StateOwner.ARTIFACT: self.artifact_root,
            StateOwner.LEGACY: self.legacy_root,
        }[owner]
        return root.joinpath(*coordinate.parts())

    def run_staging(self, *, domain: str, suite: str, study: str, variant: str, run_key: str) -> Path:
        _validate_parts((domain, suite, study, variant, run_key))
        return self.result_staging_root / domain / suite / study / variant / run_key


def _validate_parts(parts: tuple[str, ...]) -> None:
    for part in parts:
        if not part or part in {".", ".."} or Path(part).name != part:
            raise ValueError(f"invalid state coordinate component: {part!r}")


__all__ = ["StateCoordinate", "StateOwner", "WorkspacePaths"]
