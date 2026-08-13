"""Audit durable staging mirrors and conservatively clean verified copies."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from exp.framework.paths import WorkspacePaths

from .cutover import require_cutover_safe
from .projection import (
    LEGACY_E05_ORIGINAL_SEED4_COORDINATE,
    LEGACY_E05_ORIGINAL_SEED4_RUN_ID,
)


@dataclass(frozen=True)
class StorageEntry:
    path: str
    lifecycle: str
    run_id: str | None = None
    reason: str | None = None


def audit_storage(
    client,
    paths: WorkspacePaths,
    *,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Classify every staging payload without moving or deleting files."""
    require_cutover_safe(client)
    entries = []
    root = paths.result_staging_root
    manifests = sorted(root.glob("*/*/*/*/*/record/result_manifest.json")) if root.exists() else []
    for manifest_path in manifests:
        run_root = manifest_path.parent.parent
        relative = run_root.relative_to(root)
        domain, suite, _study, variant, run_key = relative.parts
        if domain != "deepscratch":
            entries.append(StorageEntry(str(run_root), "unaudited", reason="unknown domain"))
            continue
        valid, reason = _verify_local(manifest_path)
        if not valid:
            entries.append(StorageEntry(str(run_root), "incomplete", reason=reason))
            continue
        experiment = client.get_experiment_by_name(f"deepscratch.{suite}")
        if experiment is None:
            entries.append(StorageEntry(str(run_root), "orphan", reason="canonical namespace is absent"))
            continue
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string=(
                f"tags.`run.key` = '{_filter(run_key)}' AND "
                f"tags.`implementation.variant` = '{_filter(variant)}'"
            ),
            order_by=["attributes.start_time DESC"],
            max_results=100,
        )
        durable = next(
            (run for run in runs if run.info.status == "FINISHED" and run.data.tags.get("result.durable_complete") == "true"),
            None,
        )
        if durable is None:
            entries.append(StorageEntry(str(run_root), "incomplete", reason="no durable FINISHED MLflow run"))
        else:
            entries.append(StorageEntry(str(run_root), "verified", run_id=durable.info.run_id))
    retired = _retired_local_roots((repository_root or Path.cwd()).resolve())
    return {
        "root": str(root),
        "entries": [asdict(item) for item in entries],
        "legacy_entries": [asdict(item) for item in retired],
        "counts": {
            state: sum(item.lifecycle == state for item in entries)
            for state in ("verified", "incomplete", "orphan", "unaudited")
        },
        "legacy_fixtures": [_audit_e05_original_fixture(client)],
    }


def cleanup_verified_mirrors(
    client,
    paths: WorkspacePaths,
    *,
    apply: bool = False,
) -> dict[str, object]:
    report = audit_storage(client, paths)
    candidates = [
        Path(item["path"]) for item in report["entries"]
        if item["lifecycle"] == "verified"
    ]
    removed = []
    if apply:
        for candidate in candidates:
            candidate.relative_to(paths.result_staging_root)
            shutil.rmtree(candidate)
            removed.append(str(candidate))
    return {
        **report,
        "dry_run": not apply,
        "candidates": [str(item) for item in candidates],
        "removed": removed,
    }


def _verify_local(manifest_path: Path) -> tuple[bool, str | None]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["files"]
        root = manifest_path.parent
        for item in files:
            path = root / item["path"]
            if not path.is_file():
                return False, f"manifest file is absent: {item['path']}"
            if path.stat().st_size != int(item["size"]):
                return False, f"manifest size mismatch: {item['path']}"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != item["sha256"]:
                return False, f"manifest digest mismatch: {item['path']}"
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, f"invalid result manifest: {exc}"
    return True, None


def _retired_local_roots(repository_root: Path) -> list[StorageEntry]:
    candidates = (
        repository_root / "exp/deepscratch.ds1/results",
        repository_root / "exp/deepscratch.ds2/results",
        repository_root / "exp/deepscratch/ds1/original/legacy_results/fixed_seed",
        repository_root / "exp/deepscratch/ds2/original/legacy_results/fixed_seed",
    )
    entries = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        file_count = sum(path.is_file() for path in candidate.rglob("*"))
        entries.append(StorageEntry(
            str(candidate),
            "legacy-only",
            reason=(
                f"retired local result root ({file_count} files); "
                "read-only and never a cleanup candidate"
            ),
        ))
    return entries


def _filter(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _audit_e05_original_fixture(client) -> dict[str, object]:
    """Audit the preserved DS2 e05 original run without copying it."""
    run_id = LEGACY_E05_ORIGINAL_SEED4_RUN_ID
    try:
        run = client.get_run(run_id)
    except Exception:
        return {"run_id": run_id, "status": "absent"}
    tags = run.data.tags
    params = run.data.params
    actual = {
        "study_id": tags.get("experiment.id") or tags.get("experiment.ids", "").split(",")[0],
        "condition_id": tags.get("condition.id") or tags.get("atomic_run.id"),
        "seed": tags.get("master_seed") or params.get("seed/master") or params.get("seed"),
    }
    if actual != LEGACY_E05_ORIGINAL_SEED4_COORDINATE:
        return {
            "run_id": run_id,
            "status": "identity-mismatch",
            "expected": LEGACY_E05_ORIGINAL_SEED4_COORDINATE,
            "actual": actual,
        }
    artifacts = []
    pending = [""]
    while pending:
        prefix = pending.pop()
        for item in client.list_artifacts(run_id, prefix):
            if item.is_dir:
                pending.append(item.path)
            else:
                artifacts.append(item.path)
    return {
        "run_id": run_id,
        "status": "verified-identity",
        "coordinate": actual,
        "artifact_count": len(artifacts),
        "has_checkpoint_inventory": any(
            "checkpoint" in path and path.endswith(".json") for path in artifacts
        ),
    }
