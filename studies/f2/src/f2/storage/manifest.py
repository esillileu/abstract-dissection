"""Manifest generation, artifact collection, and metric extraction for publication."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from mlflow.entities import Metric


def collect_upload_files(
    run_root: Path,
    artifacts: dict[str, Any],
    *,
    include_checkpoints: bool,
) -> set[Path]:
    """Determine files to upload explicitly from result.json artifacts, evaluation, and root metadata."""
    upload_files: set[Path] = set()

    for name in (
        "result.json",
        "config.json",
        "phrase_lineage.json",
        "large_scale_lineage.json",
    ):
        f = run_root / name
        if f.is_file():
            upload_files.add(f)

    for sub in ("environment", "data", "metadata"):
        sub_dir = run_root / sub
        if sub_dir.is_dir():
            for f in sub_dir.rglob("*"):
                if f.is_file() and not f.name.startswith("."):
                    upload_files.add(f)

    for art_key, art_val in artifacts.items():
        if not art_val:
            continue
        art_path = run_root / str(art_val)
        if not include_checkpoints and art_key == "checkpoint":
            continue
        if art_path.is_file():
            upload_files.add(art_path)
        elif art_path.is_dir():
            for f in art_path.rglob("*"):
                if (
                    f.is_file()
                    and not f.name.startswith(".")
                    and not f.name.endswith(".tmp")
                ):
                    upload_files.add(f)

    eval_dir = run_root / "evaluation"
    if eval_dir.is_dir():
        for f in eval_dir.rglob("*"):
            if f.is_file() and not f.name.startswith("."):
                upload_files.add(f)

    for pattern in ("observations.csv", "metrics.jsonl", "history.json"):
        for f in run_root.glob(pattern):
            if f.is_file():
                upload_files.add(f)

    return upload_files


def create_result_manifest(
    run_root: Path,
    upload_files: set[Path],
) -> tuple[Path, list[dict[str, Any]]]:
    """Generate result_manifest.json with SHA-256 digests and file sizes."""
    manifest_path = run_root / "result_manifest.json"
    manifest_files = []
    for path in sorted(upload_files):
        if path == manifest_path:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest_files.append(
            {
                "path": path.relative_to(run_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    manifest_payload = {"files": manifest_files, "checkpoints": []}
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, manifest_files


def read_metric_batches(
    metrics_file: Path,
    batch_size: int = 750,
) -> list[list[Metric]]:
    """Parse observations.csv into batches of MLflow Metric objects."""
    if not metrics_file.is_file():
        return []
    batches: list[list[Metric]] = []
    current_batch: list[Metric] = []
    now_ms = int(time.time() * 1000)
    with metrics_file.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            step = int(row.get("processed_tokens", 0))
            if "objective_loss" in row:
                current_batch.append(
                    Metric(
                        "train/objective_loss",
                        float(row["objective_loss"]),
                        now_ms,
                        step,
                    )
                )
            if "learning_rate" in row:
                current_batch.append(
                    Metric(
                        "train/learning_rate", float(row["learning_rate"]), now_ms, step
                    )
                )
            if "tokens_per_second" in row:
                current_batch.append(
                    Metric(
                        "runtime/tokens_per_second",
                        float(row["tokens_per_second"]),
                        now_ms,
                        step,
                    )
                )
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
        if current_batch:
            batches.append(current_batch)
    return batches
