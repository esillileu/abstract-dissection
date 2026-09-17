"""Facade helper functions for StudyAnalysisInput access and artifact resolution."""

from __future__ import annotations

from pathlib import Path

from repro_core.analysis.core import Curve, aggregate
from repro_core.context.paths import WorkspacePaths

from ._model import AnalysisRun
from ._study_input import StudyAnalysisInput


def artifact_file(
    data: StudyAnalysisInput, run: AnalysisRun, artifact_path: str
) -> Path | None:
    return data.artifact_file(run, artifact_path)


def artifact_rows(
    data: StudyAnalysisInput, run: AnalysisRun, artifact_path: str
) -> list[dict[str, str]]:
    return data.artifact_rows(run, artifact_path)


def histories_from_artifact(data: StudyAnalysisInput, runs, **kwargs):
    return data.histories_from_artifact(runs, **kwargs)


def metric_histories(data: StudyAnalysisInput, runs, metric: str):
    return data.metric_histories(runs, metric)


def curve_from_artifact(data: StudyAnalysisInput, runs, **kwargs) -> Curve:
    return aggregate(data.histories_from_artifact(runs, **kwargs))


def local_artifact_root(client, run_id: str) -> Path | None:
    """Resolve only the canonical staging location advertised by run tags."""
    run = client.get_run(run_id)
    tags = run.data.tags
    run_key = tags.get("run.key")
    required = (
        tags.get("domain.name"),
        tags.get("suite.name"),
        tags.get("experiment.id"),
        tags.get("implementation.variant"),
        run_key,
    )
    if not all(required):
        return None
    path = (
        WorkspacePaths.from_environment(Path.cwd()).run_staging(
            domain=str(required[0]),
            suite=str(required[1]),
            study=str(required[2]),
            variant=str(required[3]),
            run_key=str(required[4]),
        )
        / "record"
    )
    return path if path.is_dir() else None
