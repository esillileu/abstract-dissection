from __future__ import annotations

import sys
from pathlib import Path

from tqdm.auto import tqdm

from dlfs.analysis.input import AnalysisRun, local_artifact_root
from dlfs.analysis.normalization import normalize_declared_metric
from dlfs.analysis.summary import summary_declarations
from dlfs.identity import DeepScratchCoordinate, Variant, Volume

from .cache import _load_raw_result
from .selection import _row


def load_raw_metric_observations(
    *,
    client,
    selector,
    volume: Volume,
    variants: tuple[Variant, ...],
    selected: set[str],
    selections: list[tuple[str, object, str, dict[Variant, object]]],
    summary_metrics: object,
    raw_cache_dir: Path,
    refresh_raw: bool,
    selected_attempts: int,
    tracking_uri: str,
) -> tuple[list[dict[str, object]], dict[tuple[str, Variant], list[AnalysisRun]]]:
    print(
        f"analysis phase: loading raw metrics for {selected_attempts} run(s)",
        file=sys.stderr,
        flush=True,
    )
    rows: list[dict[str, object]] = []
    render_inputs: dict[tuple[str, Variant], list[AnalysisRun]] = {}
    with tqdm(
        total=selected_attempts,
        desc="Loading raw metrics",
        unit="run",
        file=sys.stderr,
    ) as metric_progress:
        for study_id, condition, selected_seed, attempts in selections:
            observations = {}
            for variant in variants:
                attempt = attempts[variant]
                if attempt is None:
                    continue
                declarations = tuple(
                    dict.fromkeys(
                        (
                            *condition.metrics,
                            *summary_declarations(study_id, summary_metrics),
                        )
                    )
                )
                native = _load_raw_result(
                    selector,
                    attempt,
                    volume=volume,
                    variant=variant,
                    declarations=declarations,
                    tracking_uri=tracking_uri,
                    cache_dir=raw_cache_dir,
                    refresh=refresh_raw,
                )
                coordinate = DeepScratchCoordinate(
                    volume, study_id, condition.canonical_id, variant
                )
                observations[variant] = {
                    metric.metric_id: normalize_declared_metric(
                        coordinate, native, metric
                    )
                    for metric in condition.metrics
                }
                render_inputs.setdefault((study_id, variant), []).append(
                    AnalysisRun(
                        run_id=attempt.run_id,
                        canonical_condition_id=condition.canonical_id,
                        native_condition_id=attempt.condition_id,
                        seed=selected_seed,
                        variant=variant,
                        result=native,
                        local_artifact_root=local_artifact_root(client, attempt.run_id),
                    )
                )
                metric_progress.update(1)
            if study_id in selected:
                for metric in condition.metrics:
                    rows.append(
                        _row(
                            study_id,
                            condition.canonical_id,
                            selected_seed,
                            metric,
                            attempts,
                            observations,
                        )
                    )
    return rows, render_inputs
