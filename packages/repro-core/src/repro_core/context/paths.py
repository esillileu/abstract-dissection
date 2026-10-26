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
    workspace_root: Path
    references_root: Path
    studies_root: Path

    def __init__(
        self,
        repo_root: Path,
        workspace_root: Path | None = None,
        references_root: Path | None = None,
        studies_root: Path | None = None,
        *,
        data_root: Path | None = None,
        artifacts_root: Path | None = None,
        cache_root: Path | None = None,
        staging_root: Path | None = None,
    ) -> None:
        object.__setattr__(self, "repo_root", Path(repo_root).resolve())
        if workspace_root is not None:
            ws = Path(workspace_root)
        elif staging_root is not None:
            ws = (
                Path(staging_root).parent / "workspace"
                if Path(staging_root).name == "staging"
                else Path(staging_root).parent
            )
        else:
            ws = self.repo_root / "workspace"
        object.__setattr__(self, "workspace_root", ws)
        object.__setattr__(
            self,
            "references_root",
            Path(references_root).resolve()
            if references_root is not None
            else self.repo_root / "references",
        )
        object.__setattr__(
            self,
            "studies_root",
            Path(studies_root).resolve()
            if studies_root is not None
            else self.repo_root / "studies",
        )

    @classmethod
    def from_environment(cls, repository_root: Path | None = None) -> RuntimePaths:
        root = (repository_root or find_repository_root()).resolve()
        raw_ws = os.getenv("REPRO_WORKSPACE_ROOT")
        if raw_ws:
            ws_path = Path(raw_ws)
            ws = (ws_path if ws_path.is_absolute() else root / ws_path).resolve()
        else:
            ws = root / "workspace"
        return cls(
            repo_root=root,
            workspace_root=ws,
            references_root=root / "references",
            studies_root=root / "studies",
        )

    @property
    def inputs_root(self) -> Path:
        return self.workspace_root / "inputs"

    @property
    def runs_root(self) -> Path:
        return self.workspace_root / "runs"

    @property
    def analysis_root(self) -> Path:
        return self.workspace_root / "analysis"

    @property
    def analysis_cache_root(self) -> Path:
        return self.analysis_root / "cache"

    @property
    def analysis_results_root(self) -> Path:
        return self.analysis_root / "results"

    @property
    def mlflow_cache_root(self) -> Path:
        return self.analysis_cache_root / "mlflow"

    @property
    def results_cache_root(self) -> Path:
        return self.analysis_cache_root / "results"

    @property
    def data_root(self) -> Path:
        """Alias for inputs_root for legacy callers."""
        return self.inputs_root

    @property
    def staging_root(self) -> Path:
        """Alias for runs_root for legacy callers."""
        return self.runs_root

    @property
    def cache_root(self) -> Path:
        """Alias for analysis_cache_root for legacy callers."""
        return self.analysis_cache_root

    @property
    def artifacts_root(self) -> Path:
        """Alias for analysis_results_root for legacy callers."""
        return self.analysis_results_root

    @property
    def results_root(self) -> Path:
        """Alias for analysis_results_root for legacy callers."""
        return self.analysis_results_root

    def inputs(self, domain: str, *parts: str) -> Path:
        _validate_parts((domain, *parts))
        target = self.inputs_root.joinpath(domain, *parts)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def runs(self, domain: str, suite: str, *parts: str) -> Path:
        _validate_parts((domain, suite, *parts))
        target = self.runs_root.joinpath(domain, suite, *parts)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def dataset(self, name: str) -> Path:
        _validate_parts((name,))
        target = self.inputs_root / name
        target.mkdir(parents=True, exist_ok=True)
        return target

    def reference(self, name: str) -> Path:
        _validate_parts((name,))
        return self.references_root / name

    def resolve(self, owner: StateOwner, coordinate: StateCoordinate) -> Path:
        if owner is StateOwner.RESULTS:
            _validate_parts((coordinate.domain,))
            target = self.analysis_results_root / coordinate.domain
            target.mkdir(parents=True, exist_ok=True)
            return target
        root = {
            StateOwner.STAGING: self.runs_root,
            StateOwner.CACHE: self.analysis_cache_root,
        }[owner]
        target = root.joinpath(*coordinate.parts())
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def run_staging(
        self,
        *,
        domain: str,
        suite: str,
        study: str = "",
        variant: str = "",
        run_key: str = "",
    ) -> Path:
        key = run_key or (f"{study}_{variant}" if study or variant else "default")
        _validate_parts((domain, suite, key))
        target = self.runs_root / domain / suite / key
        target.mkdir(parents=True, exist_ok=True)
        return target

    def analysis_cache(self, domain: str, *parts: str) -> Path:
        _validate_parts((domain, *parts))
        target = self.results_cache_root.joinpath(domain, *parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def domain_results(self, domain: str) -> Path:
        _validate_parts((domain,))
        target = self.analysis_results_root / domain
        target.mkdir(parents=True, exist_ok=True)
        return target

    def analysis_output(self, study: str, suite: str | None = None) -> Path:
        _validate_parts((study,))
        target = self.analysis_results_root / study
        if suite:
            _validate_parts((suite,))
            target = target / suite
        target.mkdir(parents=True, exist_ok=True)
        return target

    def mlflow_artifact_cache(
        self, tracking_key: str, run_id: str, *parts: str
    ) -> Path:
        _validate_parts((tracking_key, run_id, *parts))
        target = self.mlflow_cache_root.joinpath(tracking_key, run_id, *parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target


WorkspacePaths = RuntimePaths


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
