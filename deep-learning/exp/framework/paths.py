"""Workspace storage policy for experiment-owned state."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class StateOwner(str, Enum):
    STAGING = "staging"
    CACHE = "cache"
    RESULTS = "results"
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
    staging_root: Path
    cache_root: Path
    results_root: Path
    legacy_root: Path

    @classmethod
    def from_environment(cls, repository_root: Path) -> "WorkspacePaths":
        repository_root = repository_root.resolve()
        return cls(
            _root("EXP_STAGING_ROOT", repository_root / ".staging", repository_root),
            _root("EXP_CACHE_ROOT", repository_root / ".cache", repository_root),
            _root("EXP_RESULTS_ROOT", repository_root / "results_new", repository_root),
            _root("EXP_LEGACY_ROOT", repository_root / ".legacy", repository_root),
        )

    def resolve(self, owner: StateOwner, coordinate: StateCoordinate) -> Path:
        if owner is StateOwner.RESULTS:
            _validate_parts((coordinate.domain,))
            return self.results_root / "exp" / coordinate.domain
        root = {
            StateOwner.STAGING: self.staging_root / "exp",
            StateOwner.CACHE: self.cache_root / "exp",
            StateOwner.LEGACY: self.legacy_root / "exp",
        }[owner]
        return root.joinpath(*coordinate.parts())

    def run_staging(self, *, domain: str, suite: str, study: str, variant: str, run_key: str) -> Path:
        _validate_parts((domain, suite, study, variant, run_key))
        return self.staging_root / "exp" / domain / suite / study / variant / run_key

    def analysis_cache(self, domain: str, *parts: str) -> Path:
        _validate_parts((domain, *parts))
        return self.cache_root.joinpath("exp", domain, *parts)

    def domain_results(self, domain: str) -> Path:
        _validate_parts((domain,))
        return self.results_root / "exp" / domain

    def mlflow_artifact_cache(self, tracking_key: str, run_id: str, *parts: str) -> Path:
        _validate_parts((tracking_key, run_id, *parts))
        return self.cache_root.joinpath("mlflow_artifact", tracking_key, run_id, *parts)


def _root(name: str, default: Path, repository_root: Path) -> Path:
    value = Path(os.getenv(name, default))
    value = value if value.is_absolute() else repository_root / value
    resolved = value.resolve()
    # Relative overrides must remain under the repository. Absolute overrides
    # are explicit opt-in locations and are accepted.
    if not Path(os.getenv(name, default)).is_absolute() and not resolved.is_relative_to(repository_root):
        raise ValueError(f"{name} escapes repository root: {value}")
    return resolved


def _validate_parts(parts: tuple[str, ...]) -> None:
    for part in parts:
        if not part or part in {".", ".."} or Path(part).name != part:
            raise ValueError(f"invalid state coordinate component: {part!r}")


__all__ = ["StateCoordinate", "StateOwner", "WorkspacePaths"]
