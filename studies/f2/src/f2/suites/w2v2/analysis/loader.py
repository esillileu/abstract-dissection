"""Artifact loading and run inspection for W2V2 analysis."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any


class EvidenceMissing(ValueError):
    """An existing result cannot be verified from its stored evidence."""


def sanitize_relative_path(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("artifact path must be a relative string")
    p = PurePosixPath(value)
    if p.is_absolute() or not p.parts or ".." in p.parts or "\\" in value:
        raise ValueError("unsafe result artifact path")
    return p.as_posix()


def load_run_artifacts(record, config, variant, seed, *, load_arrays: bool = True):
    """Read the published lookup; execution lineage is owned by the executor."""
    from f2.suites.common.w2v.artifacts import load_lookup_artifact

    if record.get("error") or not record.get("artifact_root"):
        raise EvidenceMissing(record.get("error", "cached artifacts unavailable"))
    root = Path(record["artifact_root"])
    result = json.loads((root / "result.json").read_text())
    lookup_rel = sanitize_relative_path(result["artifacts"]["lookup"])
    lookup = root / lookup_rel
    if not lookup.resolve().is_relative_to(root.resolve()):
        raise ValueError("lookup escapes run cache")
    if not lookup.is_dir():
        raise EvidenceMissing(f"missing cached artifact: {lookup_rel}")
    manifest_path = lookup / "manifest.json"
    if load_arrays:
        values = load_lookup_artifact(lookup)
        manifest_data = values.manifest
    else:
        values = lookup
        manifest_data = (
            json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        )
    inventory = {}
    for name in ("metrics/observations.csv", "environment/system.json"):
        path = root / name
        if path.is_file():
            inventory[name] = {"size": path.stat().st_size}

    def metadata(name):
        path = root / name
        return json.loads(path.read_text()) if path.is_file() else {}

    dataset = metadata("data/dataset_manifest.json")
    large = metadata("large_scale_lineage.json")
    phrase = metadata("phrase_lineage.json")
    identity = result.get("identity", {})
    provenance = {
        "resource_id": dataset.get("resource_id"),
        "resource_version_id": dataset.get("resource_version_id"),
        "source_corpus_sha256": dataset.get("corpus_sha256"),
        "actual_lexical_words": dataset.get("lexical_tokens"),
        "paper_lexical_words": large.get("paper_lexical_tokens", 1_000_000_000),
        "phrase_tokens": phrase.get("token_count"),
        "training_corpus_sha256": manifest_data.get("corpus_digest"),
        "vocabulary_digest": manifest_data.get("vocabulary_digest"),
        "processed_tokens_by_epoch": [e["processed_tokens"] for e in result["epochs"]],
        "study_config_digest": identity.get("config_digest"),
        "engine_config_digest": manifest_data.get("config_digest"),
        "execution_plan_id": identity.get("execution_plan_id"),
        "experiment_spec_id": record["tags"].get("experiment_spec.id"),
        "atomic_variant": variant,
    }
    return values, result, root, inventory, provenance


def extract_timing(root, inventory, result, record=None):
    """Sum the final observed wall time from each training epoch, with MLflow fallback."""
    try:
        hardware = {}
        hw_path = root / "environment/system.json"
        if hw_path.is_file():
            try:
                hw_data = json.loads(hw_path.read_text())
                if isinstance(hw_data, dict):
                    hardware = hw_data
            except Exception:
                pass

        metrics_rel = result.get("artifacts", {}).get("metrics")
        path = (root / metrics_rel) if metrics_rel else None
        if path and path.is_file():
            endpoints = {e["epoch"] for e in result.get("epochs", [])}
            by_epoch: dict[int, float] = {}
            for row in csv.DictReader(path.open()):
                epoch = int(row["epoch"])
                elapsed = float(row["elapsed_seconds"])
                if math.isfinite(elapsed) and elapsed >= 0:
                    by_epoch[epoch] = max(by_epoch.get(epoch, 0.0), elapsed)

            if by_epoch and (not endpoints or endpoints.issubset(by_epoch.keys())):
                durations = [by_epoch[e] for e in sorted(by_epoch)]
                total_sec = sum(durations)
                linked = [
                    {
                        "epoch": e,
                        "processed_tokens": 0,
                        "elapsed_seconds": by_epoch[e],
                        "hardware": hardware,
                    }
                    for e in sorted(by_epoch)
                ]
                return {
                    "status": "complete",
                    "minutes": total_sec / 60.0,
                    "seconds": total_sec,
                    "hardware": hardware,
                    "verified_observation_count": len(durations),
                    "observations": linked,
                    "protocol": "sum per-epoch cumulative endpoint seconds",
                }
            elif by_epoch and endpoints:
                return {
                    "status": "unverified",
                    "minutes": None,
                    "reason": "partial epoch coverage",
                }
    except Exception:
        pass

    if record is not None:
        start_ms = record.get("start_time")
        end_ms = record.get("end_time")
        if (
            isinstance(start_ms, (int, float))
            and isinstance(end_ms, (int, float))
            and end_ms > start_ms
        ):
            dur_sec = (end_ms - start_ms) / 1000.0
            return {
                "status": "complete",
                "minutes": dur_sec / 60.0,
                "seconds": dur_sec,
                "hardware": {},
                "verified_observation_count": 0,
                "observations": [],
                "protocol": "mlflow run duration (end_time - start_time)",
            }

    return {"status": "unverified", "minutes": None, "reason": "no timing data"}


def metric_dict(result):
    valid = result.valid_count
    return {
        "status": "complete" if valid else "no_data",
        "total_count": result.total_count,
        "valid_count": valid,
        "correct_count": round(result.require_score() * valid) if valid else 0,
        "excluded_count": result.excluded_count,
        "coverage": result.coverage,
        "accuracy_percent": 100 * result.score if valid else None,
    }


__all__ = [
    "EvidenceMissing",
    "extract_timing",
    "load_run_artifacts",
    "metric_dict",
    "sanitize_relative_path",
]
