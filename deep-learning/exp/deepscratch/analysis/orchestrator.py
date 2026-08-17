"""The single normalized analysis path for every DeepScratch variant."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys

from mlflow.tracking import MlflowClient
from tqdm.auto import tqdm

from exp.framework.paths import StateCoordinate, StateOwner, WorkspacePaths
from exp.framework.results import ArtifactReference, MetricSeries, NativeRunResult
from mlprosection_mlflow.artifact_cache import (
    artifact_download_progress,
    tracking_uri_key,
)

from ..identity import DeepScratchCoordinate, Variant, Volume
from ..execution.selection import CanonicalAttemptSelector
from .input import AnalysisRun, StudyAnalysisInput, local_artifact_root
from .normalization import normalize_declared_metric
from .paths import result_stem
from .summary import (
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
    refresh: str | bool | None = None,
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
    selector = CanonicalAttemptSelector(client, tracking_uri=tracking_uri)
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
        selections=selections,
    )
    analysis_path = cache_dir / "analysis_input.json"
    cached_analysis = _load_analysis_cache(
        analysis_path, signature, refresh=refresh_analysis
    )
    if cached_analysis is not None:
        print("analysis cache hit; rendering cached analysis artifacts", file=sys.stderr)
        rows, render_inputs = cached_analysis
    else:
        print(
            f"analysis phase: loading raw metrics for {selected_attempts} run(s)",
            file=sys.stderr,
            flush=True,
        )
        rows = []
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
                    declarations = tuple(dict.fromkeys((
                        *condition.metrics,
                        *summary_declarations(study_id, summary_metrics),
                    )))
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
                            local_artifact_root=local_artifact_root(
                                client, attempt.run_id
                            ),
                        )
                    )
                    metric_progress.update(1)
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
        _write_analysis_cache(analysis_path, signature, rows, render_inputs)
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
            "" if seed is None and run_id is None else (
                f"_seed-{seed}" if seed is not None else f"_run-{run_id[:8]}"
            ),
            cache_dir,
            refresh_raw,
            refresh_analysis,
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
                "metrics": [
                    {
                        "metric_id": metric.metric_id,
                        "unit": metric.unit,
                        "split": metric.split,
                        "axis": metric.axis,
                        "native_ids": list(metric.native_ids(variant)),
                        "value_scale": metric.value_scale,
                    }
                    for metric in condition.metrics
                ],
            })
    return {
        "version": 3,
        "tracking_uri": tracking_uri,
        "volume": volume.value,
        "studies": sorted(selected),
        "variants": [variant.value for variant in variants],
        "seed": seed,
        "run_id": run_id,
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


def _load_analysis_cache(
    cache_path: Path,
    signature: dict[str, object],
    *,
    refresh: bool,
) -> tuple[list[dict[str, object]], dict[tuple[str, Variant], list[AnalysisRun]]] | None:
    if refresh:
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("signature") != signature:
            return None
        rows = list(payload["observations"])
        render_inputs: dict[tuple[str, Variant], list[AnalysisRun]] = {}
        for item in payload["runs"]:
            run = _analysis_run_from_dict(item)
            render_inputs.setdefault((str(item["study_id"]), run.variant), []).append(run)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return rows, render_inputs


def _write_analysis_cache(
    cache_path: Path,
    signature: dict[str, object],
    observations: list[dict[str, object]],
    render_inputs: dict[tuple[str, Variant], list[AnalysisRun]],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "signature": signature,
                "observations": observations,
                "runs": [
                    _analysis_run_to_dict(study_id, run)
                    for (study_id, _variant), runs in sorted(
                        render_inputs.items(), key=lambda item: (item[0][0], item[0][1].value)
                    )
                    for run in runs
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_raw_result(
    selector,
    attempt,
    *,
    volume,
    variant,
    declarations,
    tracking_uri: str,
    cache_dir: Path,
    refresh: bool,
) -> NativeRunResult:
    declaration_payload = [
        {
            "metric_id": metric.metric_id,
            "unit": metric.unit,
            "split": metric.split,
            "axis": metric.axis,
            "native_ids": list(metric.native_ids(variant)),
        }
        for metric in declarations
    ]
    digest = hashlib.sha256(
        json.dumps(declaration_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    path = (
        cache_dir
        / tracking_uri_key(tracking_uri)
        / attempt.run_id
        / f"native-result-{digest}.json"
    )
    if not refresh:
        try:
            return _native_result_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    result = selector.load_result(
        attempt,
        volume=volume,
        variant=variant,
        declarations=declarations,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_native_result_to_dict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _analysis_run_to_dict(study_id: str, run: AnalysisRun) -> dict[str, object]:
    return {
        "study_id": study_id,
        "run_id": run.run_id,
        "canonical_condition_id": run.canonical_condition_id,
        "native_condition_id": run.native_condition_id,
        "seed": run.seed,
        "variant": run.variant.value,
        "result": _native_result_to_dict(run.result),
        "local_artifact_root": (
            None if run.local_artifact_root is None else str(run.local_artifact_root)
        ),
    }


def _analysis_run_from_dict(item: dict[str, object]) -> AnalysisRun:
    local_root = item.get("local_artifact_root")
    return AnalysisRun(
        run_id=str(item["run_id"]),
        canonical_condition_id=str(item["canonical_condition_id"]),
        native_condition_id=str(item["native_condition_id"]),
        seed=str(item["seed"]),
        variant=Variant(str(item["variant"])),
        result=_native_result_from_dict(item["result"]),
        local_artifact_root=None if local_root is None else Path(str(local_root)),
    )


def _native_result_to_dict(result: NativeRunResult) -> dict[str, object]:
    return {
        "run_id": result.run_id,
        "schema_name": result.schema_name,
        "schema_version": result.schema_version,
        "protocol_version": result.protocol_version,
        "metrics": [
            {
                "metric_id": metric.metric_id,
                "unit": metric.unit,
                "split": metric.split,
                "axis": metric.axis,
                "steps": list(metric.steps),
                "values": list(metric.values),
            }
            for metric in result.metrics
        ],
        "artifacts": [vars(artifact) for artifact in result.artifacts],
        "artifact_aliases": dict(result.artifact_aliases),
        "provenance": dict(result.provenance),
        "provenance_ref": result.provenance_ref,
    }


def _native_result_from_dict(item: dict[str, object]) -> NativeRunResult:
    return NativeRunResult(
        run_id=str(item["run_id"]),
        schema_name=str(item["schema_name"]),
        schema_version=int(item["schema_version"]),
        protocol_version=str(item["protocol_version"]),
        metrics=tuple(
            MetricSeries(
                metric_id=str(metric["metric_id"]),
                unit=str(metric["unit"]),
                split=str(metric["split"]),
                axis=str(metric["axis"]),
                steps=tuple(metric["steps"]),
                values=tuple(float(value) for value in metric["values"]),
            )
            for metric in item.get("metrics", [])
        ),
        artifacts=tuple(
            ArtifactReference(**artifact) for artifact in item.get("artifacts", [])
        ),
        artifact_aliases=dict(item.get("artifact_aliases", {})),
        provenance=dict(item.get("provenance", {})),
        provenance_ref=item.get("provenance_ref"),
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
    refresh_raw: bool = False,
    refresh_analysis: bool = False,
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
                print(
                    f"skipping {volume.value}/{study_id}: no FINISHED "
                    f"{variant.value} runs",
                    file=sys.stderr,
                )
                continue
            data = StudyAnalysisInput(
                client,
                declaration,
                variant,
                selected_runs,
                cache_dir=artifact_cache_dir,
                tracking_uri=tracking_uri,
                refresh_raw=refresh_raw,
                prepared_cache_dir=(
                    cache_dir / "prepared" / study_id / variant.value
                ),
                refresh_analysis=refresh_analysis,
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
                shutil.move(str(rendered_path), str(cache_output))
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
            append_markdown = getattr(renderer, "MARKDOWN_APPENDERS", {}).get(
                study_id
            )
            text_reports = [
                path for path in cached_rendered if path.suffix.lower() == ".txt"
            ]
            if append_markdown is not None and text_reports:
                append_markdown(summary_path, text_reports[0], seed=seed)
            data.commit_prepared()
            outputs.append(summary_path)
    return outputs
