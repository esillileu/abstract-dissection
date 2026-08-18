from __future__ import annotations

import json
from pathlib import Path

from mlflow.tracking import MlflowClient

from exp.deepscratch.ds2.implemented.executor import ProfileExecutor
from exp.deepscratch.ds2.implemented.spec import parse_run_spec
from exp.deepscratch.ds2.analysis import e10_word2vec_profile
from exp.deepscratch.analysis.orchestrator import write_analysis
from exp.deepscratch.execution.status import inspect_plan_status
from exp.deepscratch.identity import Variant, Volume
from exp.framework.execution import RunOptions, RunSelection
from exp.framework.execution.planning import Planner
from exp.deepscratch.ds2.catalog import IMPLEMENTED


CONFIG = Path("exp/deepscratch/ds2/config/implemented/e10_ptb_word2vec_profile.yaml")


def test_e10_pf01_expands_to_atomic_single_profile_runs() -> None:
    plans = Planner(IMPLEMENTED).build(
        RunSelection(experiment_ids=("e10",)), RunOptions(device="cpu")
    )
    assert len(plans) == 14
    assert {plan.seed for plan in plans} == {None}
    assert {plan.device for plan in plans} == {"cpu"}

    spec = parse_run_spec(
        CONFIG, atomic_run_id="PF-W2V-CBOW-ORIGINAL-NS"
    )
    config = spec.to_executor_config()
    assert spec.identity.experiment_id == "e10"
    assert spec.identity.group_id == "PF01"
    assert spec.kind == "performance_profile"
    assert config["dataset"]["source_study"] == "e02"
    assert config["profiling"]["study_kind"] == "update_breakdown"
    assert config["profiling"]["condition"] == {
        "subject_variant": "original",
        "model": "cbow",
        "objective": "negative_sampling",
    }
    assert config["tracking"]["tags"]["run.type"] == "profile"
    assert config["tracking"]["tags"]["profile.subject_variant"] == "original"
    assert config["tracking"]["tags"]["result.schema.name"] == "ds2-profile"
    assert callable(ProfileExecutor().run)


def test_profile_studies_require_explicit_selection() -> None:
    plans = Planner(IMPLEMENTED).build(
        RunSelection(all_experiments=True), RunOptions(device="cpu")
    )
    assert not {"e10", "e11"} & {plan.experiment_id for plan in plans}


def test_e10_analysis_uses_implemented_profiles_and_short_labels() -> None:
    assert e10_word2vec_profile.ATOMIC_IDS
    assert all("-IMPLEMENTED-" in atomic_id for atomic_id in e10_word2vec_profile.ATOMIC_IDS)
    assert all("-ONEHOT-" not in atomic_id for atomic_id in e10_word2vec_profile.ATOMIC_IDS)
    assert "Implemented" not in e10_word2vec_profile._condition_label(
        "PF-W2V-CBOW-IMPLEMENTED-NS"
    )
    assert e10_word2vec_profile._condition_label(
        "PF-W2V-CBOW-IMPLEMENTED-NS"
    ) == "Negative\nSampling"


def test_e10_renderer_selects_durable_profile_runs(tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(uri)
    experiment_id = client.create_experiment("deepscratch.ds2")
    for index, subject in enumerate(("original", "implemented"), start=1):
        condition = f"PF-W2V-CBOW-{subject.upper()}-NS"
        run = client.create_run(
            experiment_id,
            tags={
                "run.type": "profile",
                "experiment.id": "e10",
                "atomic_run.id": condition,
                "profile.subject_variant": subject,
                "implementation.variant": "implemented",
                "protocol.version": "ds2-e10-profile-v1",
                "result.schema.name": "ds2-profile",
                "result.schema.version": "1",
                "result.durable_complete": "true",
                "runtime.device_type": "cpu",
            },
        )
        client.log_metric(run.info.run_id, "profile/update/mean_ms", 10.0 + index)
        client.log_metric(run.info.run_id, "profile/points/ok", 1.0)
        client.log_metric(run.info.run_id, "profile/update/stdev_ms", 0.1 * index)
        client.log_metric(run.info.run_id, "profile/update/cold_ms", 20.0 + index)
        client.log_metric(run.info.run_id, "profile/epoch/estimated_seconds", 100.0)
        client.log_metric(run.info.run_id, "profile/total/estimated_seconds", 1000.0)
        artifact_dir = tmp_path / condition
        artifact_dir.mkdir()
        artifact = artifact_dir / "result.json"
        artifact.write_text(json.dumps({
            "schema_name": "ds2-profile",
            "points": [{
                "condition_id": condition,
                "sections": {"modules": [{
                    "component": "model_forward",
                    "measurement_scope": "separate_model_objective",
                    "timing": {"mean_ms": 1.5, "stdev_ms": 0.1},
                }]},
            }],
        }), encoding="utf-8")
        client.log_artifact(run.info.run_id, str(artifact), "profile")
        client.set_terminated(run.info.run_id, "FINISHED")

    canonical_output = tmp_path / "canonical-results"
    canonical_cache = tmp_path / "canonical-cache"
    write_analysis(
        uri,
        volume=Volume.DS2,
        experiment_ids=["e10"],
        variants=(Variant.IMPLEMENTED,),
        output_dir=canonical_output,
        cache_dir=canonical_cache,
    )
    assert (canonical_output / "ds2_e10_imp.png").exists()
    assert (canonical_output / "ds2_e10_imp_cbow.png").exists()
    operations = canonical_cache / "render" / "ds2_e10_imp_operations.csv"
    assert "model_forward" in operations.read_text(encoding="utf-8")
    assert (canonical_cache / "analysis_input.json").exists()
    assert (canonical_cache / "prepared" / "e10" / "implemented").exists()

    plans = Planner(IMPLEMENTED).build(
        RunSelection(
            experiment_ids=("e10",),
            atomic_run_ids=("PF-W2V-CBOW-ORIGINAL-NS",),
        ),
        RunOptions(device="cpu"),
    )
    report = inspect_plan_status(
        client,
        plans,
        volume=Volume.DS2,
        variant=Variant.IMPLEMENTED,
        expected_protocols={
            ("e10", "PF-W2V-CBOW-ORIGINAL-NS"): "ds2-e10-profile-v1"
        },
    )
    assert report.counts["completed"] == 1
