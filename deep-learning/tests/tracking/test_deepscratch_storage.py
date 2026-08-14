from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mlflow.tracking import MlflowClient

from exp.deepscratch.identity import Variant, Volume
from exp.deepscratch.legacy.storage_audit import (
    audit_storage,
    cleanup_verified_mirrors,
)
from exp.deepscratch.execution.selection import CanonicalAttemptSelector
from exp.framework.paths import WorkspacePaths


def _paths(tmp_path: Path) -> WorkspacePaths:
    return WorkspacePaths(
        staging_root=tmp_path / "staging",
        cache_root=tmp_path / "cache",
        results_root=tmp_path / "results",
        legacy_root=tmp_path / "legacy",
    )


def _uri(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


def _manifest(paths: WorkspacePaths, run_key: str) -> Path:
    run = paths.run_staging(
        domain="deepscratch",
        suite="ds2",
        study="e05",
        variant="original",
        run_key=run_key,
    )
    record = run / "record"
    payload = record / "metrics/final.json"
    payload.parent.mkdir(parents=True)
    payload.write_text('{"test": 1}\n', encoding="utf-8")
    manifest = record / "result_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "files": [{
            "path": "metrics/final.json",
            "size": payload.stat().st_size,
            "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        }],
    }), encoding="utf-8")
    return run


def test_storage_cleanup_only_removes_verified_mirrors(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    run_key = "run-key"
    local = _manifest(paths, run_key)
    client = MlflowClient(_uri(tmp_path))
    experiment = client.create_experiment("deepscratch.ds2")
    run = client.create_run(experiment, tags={
        "run.type": "seed_trial",
        "run.key": run_key,
        "implementation.variant": "original",
        "result.durable_complete": "true",
    })
    client.set_terminated(run.info.run_id)

    report = audit_storage(client, paths)
    assert report["counts"]["verified"] == 1
    dry_run = cleanup_verified_mirrors(client, paths)
    assert dry_run["candidates"] == [str(local)]
    assert local.exists()
    cleanup_verified_mirrors(client, paths, apply=True)
    assert not local.exists()


def test_storage_cleanup_migrates_verified_old_results_root(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    paths = _paths(tmp_path)
    transitional = WorkspacePaths(
        staging_root=tmp_path / "old-staging",
        cache_root=paths.cache_root,
        results_root=paths.results_root,
        legacy_root=paths.legacy_root,
    )
    generated = _manifest(transitional, "old-run")
    local = (
        tmp_path / "results/experiments"
        / generated.relative_to(transitional.staging_root / "exp")
    )
    local.parent.mkdir(parents=True)
    generated.rename(local)
    client = MlflowClient(_uri(tmp_path))
    experiment = client.create_experiment("deepscratch.ds2")
    run = client.create_run(experiment, tags={
        "run.key": "old-run",
        "implementation.variant": "original",
        "result.durable_complete": "true",
    })
    client.set_terminated(run.info.run_id)

    report = cleanup_verified_mirrors(client, paths)
    assert report["candidates"] == [str(local)]
    cleanup_verified_mirrors(client, paths, apply=True)
    assert not local.exists()


def test_storage_audit_reports_retired_source_tree_results_as_read_only(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    retired = repository / "exp/deepscratch.ds2/results"
    retired.mkdir(parents=True)
    (retired / "payload.json").write_text("{}\n", encoding="utf-8")
    client = MlflowClient(_uri(tmp_path))

    report = audit_storage(
        client,
        _paths(tmp_path),
        repository_root=repository,
    )

    assert report["legacy_entries"] == [{
        "path": str(retired),
        "lifecycle": "legacy-only",
        "run_id": None,
        "reason": (
            "retired local result root (1 files); "
            "read-only and never a cleanup candidate"
        ),
    }]
    assert cleanup_verified_mirrors(client, _paths(tmp_path))["candidates"] == []


def test_selector_prefers_canonical_and_excludes_alternate(tmp_path: Path) -> None:
    client = MlflowClient(_uri(tmp_path))
    legacy = client.create_experiment("ds2_original")
    canonical = client.create_experiment("deepscratch.ds2")
    common = {
        "run.type": "seed_trial",
        "experiment.id": "e05",
        "condition.id": "BETTER-RNNLM",
        "master_seed": "4",
    }
    native = client.create_run(legacy, start_time=1, tags=common)
    client.set_terminated(native.info.run_id)
    alternate = client.create_run(legacy, start_time=3, tags={
        **common, "transfer.import.disposition": "imported-alternate"
    })
    client.set_terminated(alternate.info.run_id)
    primary = client.create_run(canonical, start_time=2, tags={
        **common,
        "implementation.variant": "original",
        "result.durable_complete": "true",
    })
    client.set_terminated(primary.info.run_id)

    selector = CanonicalAttemptSelector(client)
    selected = selector.select(
        Volume.DS2,
        Variant.ORIGINAL,
        study_id="e05",
        condition_ids=("BETTER-RNNLM",),
        seed=4,
    )
    assert selected is not None and selected.run_id == primary.info.run_id
    explicit = selector.select(
        Volume.DS2,
        Variant.ORIGINAL,
        study_id="e05",
        condition_ids=("BETTER-RNNLM",),
        seed=4,
        run_id=alternate.info.run_id,
    )
    assert explicit is not None and explicit.run_id == alternate.info.run_id


def test_selector_fetches_each_variant_inventory_only_once(tmp_path: Path) -> None:
    source = MlflowClient(_uri(tmp_path))
    source.create_experiment("deepscratch.ds2")
    source.create_experiment("ds2")

    class CountingClient:
        def __init__(self, client) -> None:
            self.client = client
            self.search_count = 0

        def search_runs(self, *args, **kwargs):
            self.search_count += 1
            return self.client.search_runs(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.client, name)

    client = CountingClient(source)
    selector = CanonicalAttemptSelector(client)

    selector.attempts(Volume.DS2, Variant.IMPLEMENTED)
    selector.attempts(Volume.DS2, Variant.IMPLEMENTED)

    assert client.search_count == 2
