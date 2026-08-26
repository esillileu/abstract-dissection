"""Central repository path resolution and runtime workspace storage policy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class StateOwner(StrEnum):
    STAGING = "staging"
    CACHE = "cache"
    RESULTS = "results"


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


def find_repository_root(start_path: Path | None = None) -> Path:
    """Locate the root of the abstract-dissection monorepo."""
    current = (start_path or Path.cwd()).resolve()
    for parent in (current, *current.parents):
        if (parent / ".git").exists() or (
            (parent / "pyproject.toml").exists() and (parent / "packages").exists()
        ):
            return parent
    return current


@dataclass(frozen=True)
class RuntimePaths:
    repo_root: Path
    data_root: Path
    artifacts_root: Path
    cache_root: Path
    staging_root: Path
    references_root: Path
    studies_root: Path

    @classmethod
    def from_environment(cls, repository_root: Path | None = None) -> RuntimePaths:
        root = (repository_root or find_repository_root()).resolve()
        return cls(
            repo_root=root,
            data_root=_resolve_root(
                ["REPRO_DATA_ROOT", "EXP_DATA_ROOT"], root / "data", root
            ),
            artifacts_root=_resolve_root(
                ["REPRO_ARTIFACTS_ROOT", "EXP_RESULTS_ROOT"], root / "artifacts", root
            ),
            cache_root=_resolve_root(
                ["REPRO_CACHE_ROOT", "EXP_CACHE_ROOT"], root / ".cache", root
            ),
            staging_root=_resolve_root(
                ["REPRO_STAGING_ROOT", "EXP_STAGING_ROOT"], root / ".staging", root
            ),
            references_root=_resolve_root(
                ["REPRO_REFERENCES_ROOT"], root / "references", root
            ),
            studies_root=root / "studies",
        )

    @property
    def results_root(self) -> Path:
        """Alias for artifacts_root for compatibility with legacy tests."""
        return self.artifacts_root

    def dataset(self, name: str) -> Path:
        _validate_parts((name,))
        target = self.data_root / name
        target.mkdir(parents=True, exist_ok=True)
        return target

    def reference(self, name: str) -> Path:
        _validate_parts((name,))
        return self.references_root / name

    def resolve(self, owner: StateOwner, coordinate: StateCoordinate) -> Path:
        if owner is StateOwner.RESULTS:
            _validate_parts((coordinate.domain,))
            return self.artifacts_root / "exp" / coordinate.domain
        root = {
            StateOwner.STAGING: self.staging_root / "exp",
            StateOwner.CACHE: self.cache_root / "exp",
        }[owner]
        return root.joinpath(*coordinate.parts())

    def run_staging(
        self, *, domain: str, suite: str, study: str, variant: str, run_key: str
    ) -> Path:
        _validate_parts((domain, suite, study, variant, run_key))
        return self.staging_root / "exp" / domain / suite / study / variant / run_key

    def analysis_cache(self, domain: str, *parts: str) -> Path:
        _validate_parts((domain, *parts))
        return self.cache_root.joinpath("exp", domain, *parts)

    def domain_results(self, domain: str) -> Path:
        _validate_parts((domain,))
        return self.artifacts_root / "exp" / domain

    def run_artifact(
        self, study: str, suite: str, experiment: str, run_id: str
    ) -> Path:
        _validate_parts((study, suite, experiment, run_id))
        target = self.artifacts_root / "runs" / study / suite / experiment / run_id
        target.mkdir(parents=True, exist_ok=True)
        return target

    def analysis_output(self, study: str, suite: str | None = None) -> Path:
        _validate_parts((study,))
        target = self.artifacts_root / "analysis" / study
        if suite:
            _validate_parts((suite,))
            target = target / suite
        target.mkdir(parents=True, exist_ok=True)
        return target

    def mlflow_artifact_cache(
        self, tracking_key: str, run_id: str, *parts: str
    ) -> Path:
        _validate_parts((tracking_key, run_id, *parts))
        return self.cache_root.joinpath("mlflow_artifact", tracking_key, run_id, *parts)


WorkspacePaths = RuntimePaths


def _resolve_root(names: list[str], default: Path, repository_root: Path) -> Path:
    raw = None
    for name in names:
        val = os.getenv(name)
        if val:
            raw = val
            break
    if raw is None:
        value = default
    else:
        value = Path(raw)
        value = value if value.is_absolute() else repository_root / value

    resolved = value.resolve()
    # Relative overrides must remain under the repository.
    if (
        raw is not None
        and not Path(raw).is_absolute()
        and not resolved.is_relative_to(repository_root)
    ):
        raise ValueError(f"{name} escapes repository root: {value}")
    return resolved


def _validate_parts(parts: tuple[str, ...]) -> None:
    for part in parts:
        if not part or part in {".", ".."} or Path(part).name != part:
            raise ValueError(f"invalid state coordinate component: {part!r}")


__all__ = [
    "RuntimePaths",
    "StateCoordinate",
    "StateOwner",
    "WorkspacePaths",
    "find_repository_root",
]
