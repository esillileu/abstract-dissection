"""Safe, report-producing maintenance for repository MLflow checkpoints."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

CHECKPOINT_ROLES = ("latest", "best")
REPORT_ROOT = Path("infra/mlflow/data/maintenance-reports")
DEFAULT_ARTIFACT_ROOT = Path("infra/mlflow/data/artifacts")
REQUIRED_HISTORY_ARTIFACTS = ("updates.csv", "evaluations.csv")
COMMON_ARTIFACTS = (
    "config/resolved.json",
    "metrics/final.json",
    "checkpoints/checkpoint_manifest.json",
)


def path_digest(path: Path) -> str:
    """Use the checkpoint manager's deterministic tree digest contract."""
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        with item.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _report(
    command: str, *, apply: bool, entries: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "command": command,
        "mode": "apply" if apply else "dry-run",
        "created_at": datetime.now(UTC).isoformat(),
        "entries": entries,
    }


def write_report(report: dict[str, Any], report_root: Path = REPORT_ROOT) -> Path:
    report_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    path = report_root / f"{stamp}-{report['command']}-{report['mode']}.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _runs(client: MlflowClient, experiment_id: str, *, active_only: bool = True):
    return list(
        client.search_runs(
            [experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY if active_only else ViewType.ALL,
            max_results=50_000,
        )
    )


def _experiment(client: MlflowClient, name: str):
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {name}")
    return experiment


def _canonical_seed_runs(runs: Iterable[Any]) -> list[Any]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for run in runs:
        if run.data.tags.get("run.type") != "seed_trial":
            continue
        key = run.data.tags.get("run.key")
        if key:
            grouped[key].append(run)
    return [
        max(group, key=lambda run: run.info.end_time or run.info.start_time or 0)
        for group in grouped.values()
    ]


def _canonical_seed_runs_by_completeness(
    client: MlflowClient,
    runs: Iterable[Any],
    *,
    experiment_name: str,
    repository_root: Path,
) -> list[Any]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for run in runs:
        if run.data.tags.get("run.type") == "seed_trial":
            key = run.data.tags.get("run.key")
            if key:
                grouped[key].append(run)
    selected = []
    for run_key, group in grouped.items():
        if len(group) == 1:
            selected.append(group[0])
            continue
        local_root = (
            repository_root / "exp" / experiment_name / "results/checkpoints" / run_key
        )
        record_root = (
            repository_root
            / "exp"
            / experiment_name
            / "results/mlflow_artifacts"
            / run_key
        )
        local_roles = resolve_local_roles(local_root, artifact_root=record_root)
        selected.append(
            max(
                group,
                key=lambda run: _completeness(
                    client, run, local_checkpoint_count=len(local_roles)
                ),
            )
        )
    return selected


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def resolve_local_roles(
    checkpoint_root: Path,
    *,
    artifact_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve v2 pointers, legacy manifests, and legacy final.npz."""
    roles: dict[str, dict[str, Any]] = {}
    for role in CHECKPOINT_ROLES:
        pointer = checkpoint_root / f"{role}.json"
        payload = _load_json(pointer)
        if payload and payload.get("path"):
            path = checkpoint_root / str(payload["path"])
            if path.exists():
                digest = path_digest(path)
                expected = payload.get("sha256")
                if expected and expected != digest:
                    raise ValueError(
                        f"checkpoint digest mismatch: {path}: {digest} != {expected}"
                    )
                roles[role] = {
                    "path": path,
                    "digest": digest,
                    "pointer": pointer,
                    "source": "v2",
                }
    manifest = (
        _load_json(artifact_root / "checkpoints/checkpoint_manifest.json")
        if artifact_root is not None
        else None
    )
    if manifest:
        for role in CHECKPOINT_ROLES:
            item = manifest.get(role)
            if role == "latest" and not item:
                item = manifest.get("final")
            if role in roles or not isinstance(item, dict) or not item.get("path"):
                continue
            path = Path(str(item["path"]))
            if not path.is_absolute():
                path = checkpoint_root / path
            if path.exists():
                digest = path_digest(path)
                expected = item.get("digest") or item.get("sha256")
                if expected and expected != digest:
                    raise ValueError(
                        f"checkpoint digest mismatch: {path}: {digest} != {expected}"
                    )
                roles[role] = {
                    "path": path,
                    "digest": digest,
                    "pointer": None,
                    "source": "legacy-manifest",
                }
    legacy_final = checkpoint_root / "final.npz"
    if "latest" not in roles and legacy_final.is_file():
        roles["latest"] = {
            "path": legacy_final,
            "digest": path_digest(legacy_final),
            "pointer": None,
            "source": "legacy-final",
        }
    return roles


def _artifact_names(client: MlflowClient, run_id: str, path: str = "") -> set[str]:
    names: set[str] = set()
    pending = [path]
    while pending:
        parent = pending.pop()
        for item in client.list_artifacts(run_id, parent):
            if item.is_dir:
                pending.append(item.path)
            else:
                names.add(item.path)
    return names


def _remote_role_digest(
    client: MlflowClient,
    run_id: str,
    role: str,
    names: set[str],
) -> tuple[str | None, str | None]:
    pointer_name = f"checkpoints/{role}.json"
    if pointer_name not in names:
        return None, None
    with TemporaryDirectory(prefix="checkpoint-pointer-") as temp:
        pointer = Path(client.download_artifacts(run_id, pointer_name, temp))
        payload = _load_json(pointer)
        if not payload or not payload.get("path"):
            return None, None
        relative = f"checkpoints/{str(payload['path']).lstrip('/')}"
        try:
            downloaded = Path(client.download_artifacts(run_id, relative, temp))
        except Exception:
            return relative, None
        return relative, path_digest(downloaded)


def checkpoint_backfill(
    tracking_uri: str,
    experiments: Iterable[str],
    *,
    apply: bool = False,
    repository_root: Path = Path("."),
) -> dict[str, Any]:
    client = MlflowClient(tracking_uri=tracking_uri)
    entries: list[dict[str, Any]] = []
    for name in experiments:
        experiment = _experiment(client, name)
        for run in _canonical_seed_runs_by_completeness(
            client,
            _runs(client, experiment.experiment_id),
            experiment_name=name,
            repository_root=repository_root,
        ):
            run_id = run.info.run_id
            run_key = run.data.tags["run.key"]
            checkpoint_root = (
                repository_root / "exp" / name / "results/checkpoints" / run_key
            )
            artifact_root = (
                repository_root / "exp" / name / "results/mlflow_artifacts" / run_key
            )
            roles = resolve_local_roles(checkpoint_root, artifact_root=artifact_root)
            remote_names = _artifact_names(client, run_id, "checkpoints")
            missing_roles: list[str] = []
            for role in CHECKPOINT_ROLES:
                local = roles.get(role)
                remote_path, remote_digest = _remote_role_digest(
                    client, run_id, role, remote_names
                )
                if local is None:
                    missing_roles.append(role)
                    entries.append(
                        {
                            "experiment": name,
                            "run_id": run_id,
                            "run_key": run_key,
                            "role": role,
                            "action": "missing",
                            "reason": "no local or MLflow checkpoint",
                        }
                    )
                    if apply:
                        client.set_tag(run_id, f"checkpoint.{role}.status", "missing")
                        client.set_tag(
                            run_id,
                            f"checkpoint.{role}.missing_reason",
                            "no local or MLflow checkpoint",
                        )
                    continue
                source = Path(local["path"])
                expected_remote = (
                    f"checkpoints/generations/{source.name}"
                    if source.is_dir()
                    else f"checkpoints/{source.name}"
                )
                payload_already_present = False
                if remote_path is None and expected_remote in remote_names:
                    with TemporaryDirectory(prefix="checkpoint-existing-") as temp:
                        downloaded = Path(
                            client.download_artifacts(run_id, expected_remote, temp)
                        )
                        remote_digest = path_digest(downloaded)
                    remote_path = expected_remote
                    payload_already_present = True
                if remote_digest is not None:
                    if remote_digest != local["digest"]:
                        raise ValueError(
                            f"refusing checkpoint overwrite for {run_id}/{role}: "
                            f"{remote_digest} != {local['digest']}"
                        )
                    action = "pointer-only" if payload_already_present else "verified"
                    if apply and payload_already_present:
                        _upload_pointer(
                            client,
                            run_id,
                            role,
                            source,
                            local["digest"],
                        )
                else:
                    if remote_path is not None:
                        raise ValueError(
                            f"remote checkpoint is incomplete: {run_id}/{remote_path}"
                        )
                    action = "upload"
                    if apply:
                        if source.is_dir():
                            client.log_artifacts(
                                run_id,
                                str(source),
                                artifact_path=f"checkpoints/generations/{source.name}",
                            )
                            pointer_path = f"generations/{source.name}"
                        else:
                            client.log_artifact(
                                run_id, str(source), artifact_path="checkpoints"
                            )
                            pointer_path = source.name
                        _upload_pointer(
                            client,
                            run_id,
                            role,
                            source,
                            local["digest"],
                            pointer_path=pointer_path,
                        )
                        client.set_tag(run_id, f"checkpoint.{role}.status", "present")
                        client.set_tag(
                            run_id, f"checkpoint.{role}.sha256", local["digest"]
                        )
                entries.append(
                    {
                        "experiment": name,
                        "run_id": run_id,
                        "run_key": run_key,
                        "role": role,
                        "action": action,
                        "path": str(local["path"]),
                        "digest": local["digest"],
                        "source": local["source"],
                    }
                )
            if apply and missing_roles:
                client.set_tag(
                    run_id, "checkpoint.missing_roles", ",".join(missing_roles)
                )
    return _report("checkpoint-backfill", apply=apply, entries=entries)


def _upload_pointer(
    client: MlflowClient,
    run_id: str,
    role: str,
    source: Path,
    digest: str,
    *,
    pointer_path: str | None = None,
) -> None:
    if pointer_path is None:
        pointer_path = f"generations/{source.name}" if source.is_dir() else source.name
    with TemporaryDirectory(prefix="checkpoint-backfill-") as temp:
        pointer = Path(temp) / f"{role}.json"
        pointer.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "role": role,
                    "path": pointer_path,
                    "sha256": digest,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        client.log_artifact(run_id, str(pointer), artifact_path="checkpoints")


def _safe_artifact_dir(
    artifact_root: Path,
    experiment_id: str,
    run_id: str,
) -> Path:
    root = artifact_root.resolve()
    candidate = (root / experiment_id / run_id / "artifacts").resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"artifact path escapes configured root: {candidate}")
    return candidate


def checkpoint_prune(
    tracking_uri: str,
    experiments: Iterable[str],
    *,
    apply: bool = False,
    repository_root: Path = Path("."),
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Any]:
    client = MlflowClient(tracking_uri=tracking_uri)
    entries: list[dict[str, Any]] = []
    for name in experiments:
        experiment = _experiment(client, name)
        for run in _canonical_seed_runs(_runs(client, experiment.experiment_id)):
            run_key = run.data.tags["run.key"]
            local_root = (
                repository_root / "exp" / name / "results/checkpoints" / run_key
            )
            local_roles = resolve_local_roles(
                local_root,
                artifact_root=repository_root
                / "exp"
                / name
                / "results/mlflow_artifacts"
                / run_key,
            )
            record_root = (
                repository_root / "exp" / name / "results/mlflow_artifacts" / run_key
            )
            metadata_paths = _normalize_local_metadata(
                local_root, record_root, local_roles, apply=apply
            )
            entries.extend(
                {
                    "scope": "local-metadata",
                    "experiment": name,
                    "run_id": run.info.run_id,
                    "path": str(path),
                    "action": "normalize",
                    "exists_before": path.exists(),
                    "exists_after": path.exists() or apply,
                }
                for path in metadata_paths
            )
            preserve_local = {
                Path(item["path"]).resolve() for item in local_roles.values()
            }
            preserve_local.update(
                path.resolve()
                for path in (
                    local_root / "latest.json",
                    local_root / "best.json",
                    local_root / "final.npz",
                )
                if path.exists()
            )
            local_candidates = list(local_root.glob("generations/*"))
            if local_root.exists():
                local_candidates.extend(
                    candidate
                    for candidate in local_root.iterdir()
                    if candidate.name != "generations"
                    and candidate.suffix != ".json"
                    and candidate.name != "final.npz"
                )
            for candidate in local_candidates:
                if candidate.resolve() in preserve_local:
                    continue
                entry = {
                    "scope": "local",
                    "experiment": name,
                    "run_id": run.info.run_id,
                    "path": str(candidate),
                    "role": candidate.name.split("-", 1)[0],
                    "digest": path_digest(candidate),
                    "action": "delete",
                    "size_bytes": _size(candidate),
                    "exists_before": True,
                    "exists_after": not apply,
                }
                entries.append(entry)
                if apply:
                    _remove(candidate)

            remote_root = _safe_artifact_dir(
                artifact_root, experiment.experiment_id, run.info.run_id
            )
            if not remote_root.exists():
                continue
            uri = str(run.info.artifact_uri)
            if uri.startswith("file:"):
                uri_path = Path(uri.removeprefix("file://")).resolve()
                if not uri_path.is_relative_to(artifact_root.resolve()):
                    raise ValueError(f"artifact URI outside configured root: {uri}")
            checkpoint_dir = remote_root / "checkpoints"
            preserve_remote = {
                checkpoint_dir / "checkpoint_manifest.json",
                checkpoint_dir / "latest.json",
                checkpoint_dir / "best.json",
                checkpoint_dir / "final.npz",
            }
            for role in CHECKPOINT_ROLES:
                payload = _load_json(checkpoint_dir / f"{role}.json")
                if payload and payload.get("path"):
                    preserve_remote.add(checkpoint_dir / str(payload["path"]))
            remote_manifest = checkpoint_dir / "checkpoint_manifest.json"
            if remote_manifest.exists():
                entries.append(
                    {
                        "scope": "mlflow-metadata",
                        "experiment": name,
                        "run_id": run.info.run_id,
                        "path": str(remote_manifest),
                        "action": "normalize",
                        "exists_before": True,
                        "exists_after": True,
                    }
                )
                if apply:
                    _normalize_manifest(remote_manifest)
            remote_candidates: list[Path] = []
            if checkpoint_dir.exists():
                generations = checkpoint_dir / "generations"
                if generations.exists():
                    remote_candidates.extend(generations.iterdir())
                remote_candidates.extend(
                    candidate
                    for candidate in checkpoint_dir.iterdir()
                    if candidate.name != "generations"
                )
            for candidate in remote_candidates:
                if candidate in preserve_remote:
                    continue
                entries.append(
                    {
                        "scope": "mlflow",
                        "experiment": name,
                        "run_id": run.info.run_id,
                        "path": str(candidate),
                        "role": candidate.name.split("-", 1)[0],
                        "digest": path_digest(candidate),
                        "action": "delete",
                        "size_bytes": _size(candidate),
                        "exists_before": True,
                        "exists_after": not apply,
                    }
                )
                if apply and candidate.exists():
                    _remove(candidate)
    return _report("checkpoint-prune", apply=apply, entries=entries)


def _normalize_local_metadata(
    checkpoint_root: Path,
    record_root: Path,
    roles: dict[str, dict[str, Any]],
    *,
    apply: bool,
) -> list[Path]:
    paths = [
        record_root / "checkpoints.csv",
        record_root / "checkpoints/checkpoint_manifest.json",
    ]
    if not apply:
        return [path for path in paths if path.exists()]
    csv_path, manifest_path = paths
    if csv_path.exists():
        from .schema_v1 import _normalize_checkpoints_csv

        role_manifests = {
            role: None
            if item is None
            else {
                "path": str(item["path"]),
                "digest": str(item["digest"]),
            }
            for role, item in ((role, roles.get(role)) for role in CHECKPOINT_ROLES)
        }
        _normalize_checkpoints_csv(csv_path, role_manifests)
    if manifest_path.exists():
        _normalize_manifest(manifest_path)
    return [path for path in paths if path.exists()]


def _normalize_manifest(path: Path) -> None:
    manifest = _load_json(path)
    if manifest is None:
        return
    latest = manifest.get("latest") or manifest.get("final")
    manifest["latest"] = latest
    manifest["final"] = latest
    manifest["best"] = manifest.get("best")
    manifest["periodic"] = []
    manifest["epoch_checkpoints"] = []
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _direct_artifact_names(
    client: MlflowClient, run_id: str, path: str = ""
) -> set[str]:
    return {item.path for item in client.list_artifacts(run_id, path)}


def _completeness(
    client: MlflowClient,
    run: Any,
    *,
    local_checkpoint_count: int = 0,
) -> tuple[int, int, int, float]:
    root = _direct_artifact_names(client, run.info.run_id)
    config = _direct_artifact_names(client, run.info.run_id, "config")
    metrics = _direct_artifact_names(client, run.info.run_id, "metrics")
    checkpoints_names = _direct_artifact_names(client, run.info.run_id, "checkpoints")
    names = root | config | metrics | checkpoints_names
    history = sum(name in names for name in REQUIRED_HISTORY_ARTIFACTS)
    common = sum(name in names for name in COMMON_ARTIFACTS)
    checkpoints = max(
        local_checkpoint_count,
        sum(f"checkpoints/{role}.json" in names for role in CHECKPOINT_ROLES),
    )
    migrated_at = run.data.tags.get("transfer.imported_at")
    try:
        recency = datetime.fromisoformat(str(migrated_at)).timestamp()
    except (TypeError, ValueError):
        recency = float(run.info.end_time or run.info.start_time or 0) / 1000
    return history, common, checkpoints, recency


def dedupe(
    tracking_uri: str,
    experiment_name: str,
    *,
    apply: bool = False,
    purge_artifacts: bool = False,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Any]:
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = _experiment(client, experiment_name)
    runs = _runs(client, experiment.experiment_id)
    entries: list[dict[str, Any]] = []
    canonical_parents: dict[str, Any] = {}
    score_cache: dict[str, tuple[int, int, int, float]] = {}

    def score(run: Any) -> tuple[int, int, int, float]:
        if run.info.run_id not in score_cache:
            score_cache[run.info.run_id] = _completeness(client, run)
        return score_cache[run.info.run_id]

    for run_type, identity in (
        ("condition_parent", "condition.group.key"),
        ("seed_trial", "run.key"),
    ):
        grouped: dict[str, list[Any]] = defaultdict(list)
        for run in runs:
            if run.data.tags.get("run.type") == run_type and run.data.tags.get(
                identity
            ):
                grouped[run.data.tags[identity]].append(run)
        for key, group in grouped.items():
            if len(group) == 1:
                if run_type == "condition_parent":
                    canonical_parents[key] = group[0]
                continue
            ranked = sorted(group, key=score, reverse=True)
            winner = ranked[0]
            if run_type == "condition_parent":
                canonical_parents[key] = winner
            if len(ranked) == 1:
                continue
            for loser in ranked[1:]:
                entries.append(
                    {
                        "run_type": run_type,
                        "identity": key,
                        "winner_run_id": winner.info.run_id,
                        "loser_run_id": loser.info.run_id,
                        "winner_score": score(winner),
                        "loser_score": score(loser),
                        "actions": ["soft-delete"]
                        + (["purge-artifacts"] if purge_artifacts else []),
                    }
                )
                if apply:
                    if loser.info.lifecycle_stage != "deleted":
                        client.delete_run(loser.info.run_id)
                    if purge_artifacts:
                        _remove(
                            _safe_artifact_dir(
                                artifact_root,
                                experiment.experiment_id,
                                loser.info.run_id,
                            )
                        )
    if apply:
        for run in _runs(client, experiment.experiment_id):
            if run.data.tags.get("run.type") != "seed_trial":
                continue
            group_key = run.data.tags.get("condition.group.key")
            parent = canonical_parents.get(group_key or "")
            if parent is None:
                continue
            for tag in ("mlflow.parentRunId", "parent.mlflow_run_id"):
                if run.data.tags.get(tag) != parent.info.run_id:
                    client.set_tag(run.info.run_id, tag, parent.info.run_id)
    return _report("dedupe", apply=apply, entries=entries)
