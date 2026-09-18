from __future__ import annotations

from repro_core.context import ExperimentContext
from repro_core.context.contracts import ExperimentResult

from .common import _artifact_root, _mapping


class ProfileExecutor:
    """Dispatch canonical profile studies through their shared result contract."""

    def run(
        self, config: dict[str, object], context: ExperimentContext
    ) -> ExperimentResult:
        from dlfs.ds2.profile.studies import resolve
        from dlfs.profile.result import to_experiment_result

        profiling = _mapping(config, "profiling")
        study = resolve(str(profiling.get("study_kind", "")))
        result = study.run(config, context)
        context.metadata["profile"] = {
            "study_id": result.study_id,
            "group_id": result.group_id,
            "study_kind": result.study_kind,
            "source_study": result.source_study,
        }
        return to_experiment_result(
            result,
            artifact_root=_artifact_root(config, context),
        )
