from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dlfs.analysis.input import AnalysisRun
from dlfs.analysis.summary import summary_declarations
from dlfs.identity import Variant, Volume
from repro_core.results import NativeRunResult
from repro_mlflow.artifact_cache import tracking_uri_key

from .serialization import (
    _analysis_run_from_dict,
    _analysis_run_to_dict,
    _native_result_from_dict,
    _native_result_to_dict,
)


def _cache_signature(
    *,
    tracking_uri: str,
    volume: Volume,
    selected: set[str],
    variants: tuple[Variant, ...],
    summary_metrics: object,
    seed: int | None,
    run_id: str | None,
    device: str | None,
    selections: list[tuple[str, object, str, dict[Variant, object]]],
) -> dict[str, object]:
    runs = []
    for study_id, condition, selected_seed, attempts in selections:
        for variant, attempt in attempts.items():
            if attempt is None:
                continue
            runs.append(
                {
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
                }
            )
    return {
        "version": 3,
        "tracking_uri": tracking_uri,
        "volume": volume.value,
        "studies": sorted(selected),
        "variants": [variant.value for variant in variants],
        "seed": seed,
        "run_id": run_id,
        "device": device,
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
) -> (
    tuple[list[dict[str, object]], dict[tuple[str, Variant], list[AnalysisRun]]] | None
):
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
            render_inputs.setdefault((str(item["study_id"]), run.variant), []).append(
                run
            )
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
                        render_inputs.items(),
                        key=lambda item: (item[0][0], item[0][1].value),
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
    volume: Volume,
    variant: Variant,
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
            return _native_result_from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
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
