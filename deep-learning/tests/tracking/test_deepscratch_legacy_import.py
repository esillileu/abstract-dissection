from __future__ import annotations

from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from exp.deepscratch.active_runs import find_active_legacy_runs, require_cutover_safe
from exp.deepscratch.identity import Variant, Volume
from exp.deepscratch.legacy_import import import_legacy_archive, inspect_archive
from exp.deepscratch.legacy_results import LegacyResultStore
from exp.analyze import analysis_scope, completed_seed_runs, mlflow_client
from mlprosection_mlflow.transfer import export_experiment


def _uri(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.resolve() / 'mlflow.db'}"


def _archive(
    tmp_path: Path,
    name: str,
    *,
    metric: float,
    run_key: str = "shared-run-key",
) -> Path:
    source_uri = _uri(tmp_path / f"source-{name}")
    client = MlflowClient(source_uri)
    experiment_id = client.create_experiment(
        "ds2",
        artifact_location=(tmp_path / f"source-{name}-artifacts").resolve().as_uri(),
    )
    parent = client.create_run(
        experiment_id,
        tags={
            "run.type": "condition_parent",
            "condition.group.key": "group-key",
            "experiment.ids": "e01",
            "atomic_run.id": "CBOW",
        },
    )
    client.set_terminated(parent.info.run_id)
    child = client.create_run(
        experiment_id,
        start_time=1_700_000_000_000 + int(metric * 100),
        tags={
            "run.type": "seed_trial",
            "run.key": run_key,
            "condition.group.key": "group-key",
            "experiment.ids": "e01",
            "atomic_run.id": "CBOW",
            "protocol.version": "legacy",
            "master_seed": "1",
            "mlflow.parentRunId": parent.info.run_id,
            "parent.mlflow_run_id": parent.info.run_id,
        },
    )
    client.log_param(child.info.run_id, "seed/master", "1")
    client.log_metric(child.info.run_id, "final/train/loss", metric, step=1)
    artifact = tmp_path / f"artifact-{name}.txt"
    artifact.write_text(f"payload {metric}\n", encoding="utf-8")
    client.log_artifact(child.info.run_id, str(artifact), artifact_path="result")
    client.set_terminated(child.info.run_id)
    return export_experiment(source_uri, "ds2", tmp_path / f"{name}.zip")


def _target(tmp_path: Path) -> tuple[str, MlflowClient, str]:
    uri = _uri(tmp_path / "target")
    client = MlflowClient(uri)
    experiment_id = client.create_experiment(
        "ds2", artifact_location=(tmp_path / "target-artifacts").resolve().as_uri()
    )
    return uri, client, experiment_id


def test_mapping_preflight_rejects_wrong_variant(tmp_path: Path) -> None:
    archive = _archive(tmp_path, "mapping", metric=1.0)
    with pytest.raises(ValueError, match="rejected-mapping"):
        inspect_archive(
            archive, volume=Volume.DS2, variant=Variant.ORIGINAL
        )


def test_import_is_idempotent_and_existing_state_is_immutable(tmp_path: Path) -> None:
    archive = _archive(tmp_path, "primary", metric=1.0)
    uri, client, experiment_id = _target(tmp_path)
    client.set_experiment_tag(experiment_id, "destination", "unchanged")

    first = import_legacy_archive(
        uri, archive, volume=Volume.DS2, variant=Variant.IMPLEMENTED
    )
    child = next(
        entry for entry in first["entries"] if entry["run_type"] == "seed_trial"
    )
    assert child["disposition"] == "imported"
    imported = client.get_run(child["target_run_id"])
    original_tags = dict(imported.data.tags)
    original_metrics = dict(imported.data.metrics)

    second = import_legacy_archive(
        uri, archive, volume=Volume.DS2, variant=Variant.IMPLEMENTED
    )
    second_child = next(
        entry for entry in second["entries"] if entry["run_type"] == "seed_trial"
    )
    assert second_child["disposition"] == "reused-identical"
    assert second_child["target_run_id"] == child["target_run_id"]
    assert client.get_experiment(experiment_id).tags == {"destination": "unchanged"}
    reused = client.get_run(child["target_run_id"])
    assert reused.data.tags == original_tags
    assert reused.data.metrics == original_metrics


def test_nonidentical_payload_becomes_noncanonical_alternate(tmp_path: Path) -> None:
    primary = _archive(tmp_path, "primary", metric=1.0)
    alternate = _archive(tmp_path, "alternate", metric=2.0)
    uri, client, _ = _target(tmp_path)
    first = import_legacy_archive(
        uri, primary, volume=Volume.DS2, variant=Variant.IMPLEMENTED
    )
    second = import_legacy_archive(
        uri, alternate, volume=Volume.DS2, variant=Variant.IMPLEMENTED
    )
    primary_child = next(e for e in first["entries"] if e["run_type"] == "seed_trial")
    alternate_child = next(e for e in second["entries"] if e["run_type"] == "seed_trial")
    assert alternate_child["disposition"] == "imported-alternate"
    store = LegacyResultStore(client)
    selected = store.select_attempt(
        Volume.DS2,
        Variant.IMPLEMENTED,
        experiment_id="e01",
        condition_id="CBOW",
        seed=1,
    )
    assert selected is not None
    assert selected.run_id == primary_child["target_run_id"]
    explicit = store.select_attempt(
        Volume.DS2,
        Variant.IMPLEMENTED,
        experiment_id="e01",
        condition_id="CBOW",
        seed=1,
        run_id=alternate_child["target_run_id"],
    )
    assert explicit is not None
    assert explicit.run_id == alternate_child["target_run_id"]


def test_running_collision_is_deferred_and_cutover_gate_reports_it(tmp_path: Path) -> None:
    archive = _archive(tmp_path, "deferred", metric=1.0)
    uri, client, experiment_id = _target(tmp_path)
    active = client.create_run(
        experiment_id,
        tags={
            "run.type": "seed_trial",
            "run.key": "shared-run-key",
            "experiment.ids": "e01",
            "atomic_run.id": "CBOW",
        },
    )
    report = import_legacy_archive(
        uri, archive, volume=Volume.DS2, variant=Variant.IMPLEMENTED
    )
    child = next(entry for entry in report["entries"] if entry["run_type"] == "seed_trial")
    assert child["disposition"] == "deferred-running"
    assert child["target_run_id"] == active.info.run_id
    assert len(find_active_legacy_runs(client)) == 1
    with pytest.raises(RuntimeError, match="cutover blocked"):
        require_cutover_safe(client)


def test_analysis_scope_reads_new_and_legacy_namespaces_as_one_coordinate(
    tmp_path: Path,
) -> None:
    uri = _uri(tmp_path / "analysis")
    client = MlflowClient(uri)
    legacy_id = client.create_experiment("ds2")
    new_id = client.create_experiment("deepscratch.ds2")

    def create(experiment_id: str, run_id: str, start: int, **tags: str) -> str:
        run = client.create_run(
            experiment_id,
            start_time=start,
            run_name=run_id,
            tags={
                "run.type": "seed_trial",
                "execution_group.id": "GT01",
                "atomic_run.id": "W2V-TOY-CBOW-FULL",
                "protocol.version": "legacy",
                "master_seed": "1",
                **tags,
            },
        )
        client.log_param(run.info.run_id, "seed/master", "1")
        client.set_terminated(run.info.run_id)
        return run.info.run_id

    create(legacy_id, "legacy", 10)
    newest = create(
        new_id,
        "new-implemented",
        30,
        **{"implementation.variant": "implemented"},
    )
    create(
        new_id,
        "new-original",
        40,
        **{"implementation.variant": "original"},
    )
    create(
        legacy_id,
        "alternate",
        50,
        **{"transfer.import.disposition": "imported-alternate"},
    )

    with analysis_scope(
        experiment_aliases={"ds2": ("deepscratch.ds2", "ds2")},
        variant="implemented",
    ):
        grouped = completed_seed_runs(
            mlflow_client(uri),
            experiment_name="ds2",
            group_id="GT01",
            atomic_run_ids=["W2V-TOY-CBOW-FULL"],
            protocol_version="legacy",
        )

    assert [run.run_id for run in grouped["W2V-TOY-CBOW-FULL"]] == [newest]


def test_analysis_scope_falls_back_to_legacy_when_new_namespace_is_absent(
    tmp_path: Path,
) -> None:
    uri = _uri(tmp_path / "legacy-analysis")
    client = MlflowClient(uri)
    experiment_id = client.create_experiment("ds1_original")
    run = client.create_run(
        experiment_id,
        tags={
            "run.type": "seed_trial",
            "execution_group.id": "GT01",
            "atomic_run.id": "ORIGINAL",
            "master_seed": "1",
        },
    )
    client.set_terminated(run.info.run_id)

    with analysis_scope(
        experiment_aliases={
            "ds1_original": ("deepscratch.ds1", "ds1_original")
        },
        variant="original",
    ):
        grouped = completed_seed_runs(
            mlflow_client(uri),
            experiment_name="ds1_original",
            group_id="GT01",
            atomic_run_ids=["ORIGINAL"],
        )

    assert [item.run_id for item in grouped["ORIGINAL"]] == [run.info.run_id]
