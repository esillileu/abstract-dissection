from __future__ import annotations

import importlib
import sys
from pathlib import Path

from mlflow.tracking import MlflowClient

from dlfs.execution.selection import CanonicalAttemptSelector
from dlfs.identity import Variant, Volume
from dlfs.tracking import resolve_tracking_uri
from repro_core.context.paths import StateCoordinate, StateOwner, WorkspacePaths
from repro_mlflow.artifact_cache import artifact_download_progress

from .cache import (
    _cache_signature,
    _load_analysis_cache,
    _write_analysis_cache,
)
from .metrics import load_raw_metric_observations
from .rendering import _render_studies
from .selection import (
    _canonical_device,
    _seed_key,
    _seeds,
    _write_observations_csv,
)


def write_analysis(
    tracking_uri: str,
    *,
    volume: Volume,
    experiment_ids: list[str],
    variants: tuple[Variant, ...],
    output_dir: Path,
    cache_dir: Path | None = None,
    seed: int | None = None,
    device: str | None = None,
    run_id: str | None = None,
    error_style: str = "band",
    print_summary: bool = False,
    refresh: str | bool | None = None,
    artifact_cache_dir: Path | None = None,
) -> Path:
    tracking_uri = resolve_tracking_uri(tracking_uri)
    if cache_dir is None:
        cache_dir = WorkspacePaths.from_environment(Path.cwd()).resolve(
            StateOwner.CACHE,
            StateCoordinate(
                "deepscratch",
                volume.value,
                experiment_ids[0] if len(experiment_ids) == 1 else "all",
                "-".join(variant.value for variant in variants),
                "analysis",
            ),
        )
    client = MlflowClient(tracking_uri=tracking_uri)
    if artifact_cache_dir is None:
        artifact_cache_dir = (
            WorkspacePaths.from_environment(Path.cwd()).cache_root / "mlflow_artifact"
        )
    # Keep direct Python callers using the former boolean API compatible.
    if refresh is True:
        refresh = "all"
    elif refresh is False:
        refresh = None
    if refresh not in {None, "analysis", "all"}:
        raise ValueError("refresh must be None, 'analysis', or 'all'")
    refresh_analysis = refresh in {"analysis", "all"}
    refresh_raw = refresh == "all"
    raw_cache_dir = (
        WorkspacePaths.from_environment(Path.cwd()).cache_root / "mlflow_raw"
    )
    selector = CanonicalAttemptSelector(
        client,
        tracking_uri=tracking_uri,
    )
    studies = importlib.import_module(f"dlfs.{volume.value}.result_schema")
    summary_metrics = studies.SUMMARY_METRICS
    studies = studies.STUDIES
    renderer = importlib.import_module(f"dlfs.{volume.value}.analysis.render")
    selected = set(experiment_ids) if experiment_ids else set(renderer.RENDERERS)
    unsupported = selected - set(renderer.RENDERERS)
    if unsupported:
        raise ValueError("unsupported analyses: " + ", ".join(sorted(unsupported)))
    if variants == (Variant.ORIGINAL,):
        configured = {
            study_id
            for study_id in selected
            if any(
                studies[source].conditions
                and any(
                    condition.aliases(Variant.ORIGINAL)
                    for condition in studies[source].conditions
                )
                for source in renderer.STUDY_SOURCES.get(study_id, (study_id,))
            )
        }
        excluded = sorted(selected - configured)
        if excluded:
            print(
                "analysis phase: excluding studies without original configuration: "
                + ", ".join(excluded),
                file=sys.stderr,
                flush=True,
            )
        selected = configured
    source_studies = {
        source
        for study_id in selected
        for source in renderer.STUDY_SOURCES.get(study_id, (study_id,))
    }
    print(
        f"analysis phase: selecting FINISHED runs for {len(source_studies)} study(s)",
        file=sys.stderr,
        flush=True,
    )
    selections = []
    for study_id in sorted(source_studies):
        study = studies[study_id]
        for condition in study.conditions:
            seeds = _seeds(selector, volume, study_id, condition, variants)
            if seed is not None:
                seeds &= {str(seed)}
            for selected_seed in sorted(seeds, key=_seed_key):
                attempts = {}
                for variant in variants:
                    aliases = condition.aliases(variant)
                    attempt = selector.select(
                        volume,
                        variant,
                        study_id=study_id,
                        condition_ids=aliases,
                        seed=selected_seed,
                        run_id=run_id if len(variants) == 1 else None,
                        device=_canonical_device(
                            volume,
                            variant,
                            study_id=study_id,
                            condition_ids=aliases,
                        )
                        if device is None
                        else device,
                    )
                    attempts[variant] = attempt
                selections.append((study_id, condition, selected_seed, attempts))

    selected_attempts = sum(
        attempt is not None
        for _study_id, _condition, _seed, attempts in selections
        for attempt in attempts.values()
    )
    print(
        f"analysis phase: selected {selected_attempts} run(s); checking analysis cache",
        file=sys.stderr,
        flush=True,
    )

    signature = _cache_signature(
        tracking_uri=tracking_uri,
        volume=volume,
        selected=selected,
        variants=variants,
        summary_metrics=summary_metrics,
        seed=seed,
        run_id=run_id,
        device=device,
        selections=selections,
    )
    analysis_path = cache_dir / "analysis_input.json"
    cached_analysis = _load_analysis_cache(
        analysis_path, signature, refresh=refresh_analysis
    )
    if cached_analysis is not None:
        print(
            "analysis cache hit; rendering cached analysis artifacts", file=sys.stderr
        )
        rows, render_inputs = cached_analysis
    else:
        rows, render_inputs = load_raw_metric_observations(
            client=client,
            selector=selector,
            volume=volume,
            variants=variants,
            selected=selected,
            selections=selections,
            summary_metrics=summary_metrics,
            raw_cache_dir=raw_cache_dir,
            refresh_raw=refresh_raw,
            selected_attempts=selected_attempts,
            tracking_uri=tracking_uri,
        )
        _write_analysis_cache(analysis_path, signature, rows, render_inputs)
    _write_observations_csv(cache_dir / "observations.csv", rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "analysis phase: loading artifacts and rendering studies",
        file=sys.stderr,
        flush=True,
    )
    with artifact_download_progress():
        _render_studies(
            client,
            output_dir,
            artifact_cache_dir,
            tracking_uri,
            studies,
            sorted(selected),
            variants,
            render_inputs,
            volume,
            error_style,
            summary_metrics,
            print_summary,
            seed,
            ""
            if seed is None and run_id is None
            else (f"_seed-{seed}" if seed is not None else f"_run-{run_id[:8]}"),
            cache_dir,
            refresh_raw,
            refresh_analysis,
        )
    return output_dir
