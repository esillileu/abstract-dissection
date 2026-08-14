"""The single normalized analysis path for every DeepScratch variant."""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
import sys

from mlflow.tracking import MlflowClient

from exp.framework.paths import StateCoordinate, StateOwner, WorkspacePaths

from ..identity import DeepScratchCoordinate, Variant, Volume
from ..execution.selection import CanonicalAttemptSelector
from .input import AnalysisRun, StudyAnalysisInput, local_artifact_root
from .normalization import normalize_declared_metric
from .paths import result_stem
from .summary import (
    print_summary_file,
    summary_declarations,
    write_study_summary,
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
    run_id: str | None = None,
    error_style: str = "band",
    print_summary: bool = False,
    refresh: bool = False,
    artifact_cache_dir: Path | None = None,
) -> Path:
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
            WorkspacePaths.from_environment(Path.cwd()).cache_root
            / "mlflow_artifact"
        )
    selector = CanonicalAttemptSelector(client)
    studies = importlib.import_module(
        f"exp.deepscratch.{volume.value}.result_schema"
    )
    summary_metrics = studies.SUMMARY_METRICS
    studies = studies.STUDIES
    renderer = importlib.import_module(
        f"exp.deepscratch.{volume.value}.analysis.render"
    )
    selected = set(experiment_ids) if experiment_ids else set(renderer.RENDERERS)
    unsupported = selected - set(renderer.RENDERERS)
    if unsupported:
        raise ValueError(
            "unsupported analyses: " + ", ".join(sorted(unsupported))
        )
    source_studies = {
        source
        for study_id in selected
        for source in renderer.STUDY_SOURCES.get(study_id, (study_id,))
    }
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
                    attempt = selector.select(
                        volume,
                        variant,
                        study_id=study_id,
                        condition_ids=condition.aliases(variant),
                        seed=selected_seed,
                        run_id=run_id if len(variants) == 1 else None,
                    )
                    attempts[variant] = attempt
                selections.append((study_id, condition, selected_seed, attempts))

    signature = _cache_signature(
        tracking_uri=tracking_uri,
        volume=volume,
        selected=selected,
        variants=variants,
        summary_metrics=summary_metrics,
        seed=seed,
        run_id=run_id,
        error_style=error_style,
        selections=selections,
    )
    manifest_path = cache_dir / "analysis_manifest.json"
    cached_outputs = _cached_outputs(
        manifest_path,
        signature,
        output_dir,
        refresh=refresh,
    )
    if cached_outputs is not None:
        print("analysis cache hit", file=sys.stderr)
        if print_summary:
            for path in cached_outputs:
                if path.suffix == ".md":
                    print_summary_file(path)
        return output_dir

    rows = []
    render_inputs: dict[tuple[str, Variant], list[AnalysisRun]] = {}
    for study_id, condition, selected_seed, attempts in selections:
        observations = {}
        for variant in variants:
            attempt = attempts[variant]
            if attempt is None:
                continue
            native = selector.load_result(
                attempt,
                volume=volume,
                variant=variant,
                declarations=tuple(dict.fromkeys((
                    *condition.metrics,
                    *summary_declarations(study_id, summary_metrics),
                ))),
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
                    local_artifact_root=local_artifact_root(
                        client, attempt.run_id
                    ),
                )
            )
        if study_id in selected:
            for metric in condition.metrics:
                rows.append(_row(
                    study_id,
                    condition.canonical_id,
                    selected_seed,
                    metric,
                    attempts,
                    observations,
                ))
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / "observations.csv"
    fieldnames = (
        "study_id", "experiment_id", "canonical_condition_id", "condition_id",
        "seed", "metric_id", "unit", "split", "axis", "protocol",
        "protocol_version", "implemented_run_id", "implemented_value",
        "implemented_availability", "implemented_unavailable_reason",
        "implemented_native_schema", "implemented_provenance_ref",
        "original_run_id", "original_value", "original_availability",
        "original_unavailable_reason", "original_native_schema",
        "original_provenance_ref",
    )
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    visible_outputs = _render_studies(
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
        "" if seed is None and run_id is None else (
            f"_seed-{seed}" if seed is not None else f"_run-{run_id[:8]}"
        ),
        cache_dir,
    )
    _write_cache_manifest(
        manifest_path,
        signature,
        output_dir,
        visible_outputs,
        observations=output,
    )
    return output_dir


def _cache_signature(
    *,
    tracking_uri,
    volume,
    selected,
    variants,
    summary_metrics,
    seed,
    run_id,
    error_style,
    selections,
) -> dict[str, object]:
    runs = []
    for study_id, condition, selected_seed, attempts in selections:
        for variant, attempt in attempts.items():
            if attempt is None:
                continue
            runs.append({
                "study_id": study_id,
                "condition_id": condition.canonical_id,
                "seed": selected_seed,
                "variant": variant.value,
                "run_id": attempt.run_id,
            })
    return {
        "version": 2,
        "tracking_uri": tracking_uri,
        "volume": volume.value,
        "studies": sorted(selected),
        "variants": [variant.value for variant in variants],
        "seed": seed,
        "run_id": run_id,
        "error_style": error_style,
        "summary_metrics": {
            study_id: [
                {
                    "metric_id": metric.metric_id,
                    "unit": metric.unit,
                    "split": metric.split,
                    "axis": metric.axis,
                    "implemented_native_ids": list(metric.implemented_native_ids),
                    "original_native_ids": list(metric.original_native_ids),
                    "value_scale": metric.value_scale,
                }
                for metric in summary_declarations(study_id, summary_metrics)
            ]
            for study_id in sorted(selected)
        },
        "runs": runs,
    }


def _cached_outputs(
    manifest_path: Path,
    signature: dict[str, object],
    output_dir: Path,
    *,
    refresh: bool,
) -> list[Path] | None:
    if refresh:
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") != signature:
            return None
        observations = Path(manifest["observations"])
        outputs = [output_dir / item for item in manifest["outputs"]]
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        return None
    if not observations.is_file() or not outputs or not all(
        path.is_file() for path in outputs
    ):
        return None
    return outputs


def _write_cache_manifest(
    manifest_path: Path,
    signature: dict[str, object],
    output_dir: Path,
    outputs: list[Path],
    *,
    observations: Path,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "signature": signature,
                "observations": str(observations.resolve()),
                "outputs": [
                    str(path.resolve().relative_to(output_dir.resolve()))
                    for path in outputs
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _seeds(selector, volume, study_id, condition, variants) -> set[str]:
    seeds = set()
    for variant in variants:
        aliases = set(condition.aliases(variant))
        seeds.update(
            item.seed for item in selector.attempts(volume, variant)
            if item.study_id == study_id
            and item.condition_id in aliases
            and item.status == "FINISHED"
            and item.disposition != "imported-alternate"
        )
    return seeds


def _row(study_id, condition_id, seed, metric, attempts, observations):
    protocols = {
        attempt.protocol_version for attempt in attempts.values()
        if attempt is not None
    }
    if protocols and protocols <= set(metric.protocols):
        protocol = metric.protocols[0]
    else:
        protocol = "" if not protocols else "protocol-mismatch"
    row = {
        "study_id": study_id,
        "experiment_id": study_id,
        "canonical_condition_id": condition_id,
        "condition_id": condition_id,
        "seed": seed,
        "metric_id": metric.metric_id,
        "unit": metric.unit,
        "split": metric.split,
        "axis": metric.axis,
        "protocol": protocol,
        "protocol_version": protocol,
    }
    for variant in Variant:
        attempt = attempts.get(variant)
        observation = observations.get(variant, {}).get(metric.metric_id)
        available = observation is not None and observation.available
        row.update({
            f"{variant.value}_run_id": "" if attempt is None else attempt.run_id,
            f"{variant.value}_value": "" if not available else observation.values[-1],
            f"{variant.value}_availability": "available" if available else "unavailable",
            f"{variant.value}_unavailable_reason": (
                "run is absent" if attempt is None else observation.unavailable_reason
            ),
            f"{variant.value}_native_schema": "" if observation is None else observation.native_schema,
            f"{variant.value}_provenance_ref": "" if observation is None else observation.provenance_ref,
        })
    return row


def _seed_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):020d}") if value.isdigit() else (1, value)


def _render_studies(
    client,
    output_dir: Path,
    artifact_cache_dir: Path,
    tracking_uri: str,
    studies,
    selected_studies: list[str],
    variants: tuple[Variant, ...],
    render_inputs: dict[tuple[str, Variant], list[AnalysisRun]],
    volume: Volume,
    error_style: str,
    summary_metrics,
    print_summary: bool,
    seed: int | None,
    filename_suffix: str,
    cache_dir: Path,
) -> list[Path]:
    renderer = importlib.import_module(
        f"exp.deepscratch.{volume.value}.analysis.render"
    )
    outputs = []
    for study_id in selected_studies:
        if study_id not in renderer.RENDERERS:
            continue
        sources = renderer.STUDY_SOURCES.get(study_id, (study_id,))
        declaration = (
            studies[study_id]
            if sources == (study_id,)
            else type(studies[sources[0]])(
                study_id,
                tuple(
                    condition
                    for source in sources
                    for condition in studies[source].conditions
                ),
            )
        )
        for variant in variants:
            selected_runs = tuple(
                run
                for source in sources
                for run in render_inputs.get((source, variant), ())
            )
            if not selected_runs:
                if len(variants) == 1:
                    raise ValueError(
                        f"no FINISHED {variant.value} runs for "
                        f"{volume.value}/{study_id}"
                    )
                continue
            data = StudyAnalysisInput(
                client,
                declaration,
                variant,
                selected_runs,
                cache_dir=artifact_cache_dir,
                tracking_uri=tracking_uri,
            )
            output_variants = variants if len(variants) == 1 else (variant,)
            output = output_dir / (
                f"{result_stem(volume, study_id, output_variants)}{filename_suffix}.png"
            )
            rendered = renderer.render_study(
                data,
                study_id,
                output,
                error_style=error_style,
            )
            cached_rendered = []
            for rendered_path in rendered:
                if rendered_path.suffix.lower() in {".png", ".md"}:
                    outputs.append(rendered_path)
                    continue
                cache_output = cache_dir / "render" / rendered_path.name
                cache_output.parent.mkdir(parents=True, exist_ok=True)
                rendered_path.replace(cache_output)
                cached_rendered.append(cache_output)
            summary_path = write_study_summary(
                data,
                volume=volume,
                study_id=study_id,
                metrics=summary_declarations(study_id, summary_metrics),
                output_dir=output_dir,
                output_variants=output_variants,
                print_console=print_summary,
                filename_suffix=filename_suffix,
                cache_dir=cache_dir,
            )
            append_markdown = getattr(renderer, "append_markdown_report", None)
            text_reports = [
                path for path in cached_rendered if path.suffix.lower() == ".txt"
            ]
            if append_markdown is not None and text_reports:
                append_markdown(summary_path, text_reports[0], seed=seed)
            outputs.append(summary_path)
    return outputs
