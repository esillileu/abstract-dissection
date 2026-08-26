from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

from dlfs.analysis.orchestrator import write_analysis
from dlfs.ds2.analysis import e11_vocabulary_size_scaling
from dlfs.ds2.catalog import IMPLEMENTED
from dlfs.ds2.implemented.spec import parse_run_spec
from dlfs.identity import Variant, Volume
from repro_core.execution import RunOptions, RunSelection
from repro_core.execution.planning import Planner

CONFIG = Path(
    "studies/dlfs/src/dlfs/ds2/config/implemented/"
    "e11_word2vec_vocabulary_size_scaling.yaml"
)


def test_e11_pf02_declares_six_atomic_scaling_runs() -> None:
    plans = Planner(IMPLEMENTED).build(
        RunSelection(experiment_ids=("e11",)), RunOptions(device="cpu")
    )
    assert len(plans) == 6
    assert {plan.seed for plan in plans} == {None}
    spec = parse_run_spec(CONFIG, atomic_run_id="PF-VSCALE-CBOW-FUSED-NS")
    config = spec.to_executor_config()
    assert spec.identity.group_id == "PF02"
    assert config["dataset"]["source_study"] == "e02"
    assert config["profiling"]["study_kind"] == "axis_scaling"
    assert config["profiling"]["condition"] == {
        "subject_variant": "implemented",
        "model": "cbow",
        "objective": "fused_negative_sampling",
    }
    assert config["profiling"]["axis"] == {
        "name": "vocabulary_size",
        "values": "device_default",
        "schedule": "device_default-v1",
        "reverse": False,
    }
    assert config["tracking"]["tags"]["run.type"] == "profile"
    assert config["tracking"]["tags"]["profile.study_kind"] == "axis_scaling"


def test_e11_renderer_uses_a_linear_y_axis() -> None:
    figure, axis = plt.subplots()
    try:
        e11_vocabulary_size_scaling._plot_model(axis, [], "CBOW")
        assert axis.get_xscale() == "log"
        assert axis.get_yscale() == "linear"
    finally:
        plt.close(figure)


def test_e11_renderer_uses_vocabulary_size_as_x_axis(tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(uri)
    experiment_id = client.create_experiment("deepscratch.ds2")
    for condition in ("PF-VSCALE-CBOW-FS", "PF-VSCALE-CBOW-NS"):
        run = client.create_run(
            experiment_id,
            tags={
                "run.type": "profile",
                "experiment.id": "e11",
                "atomic_run.id": condition,
                "implementation.variant": "implemented",
                "result.durable_complete": "true",
                "result.schema.name": "ds2-profile",
                "result.schema.version": "1",
                "protocol.version": "ds2-e11-vocabulary-size-scaling-v1",
            },
        )
        for vocabulary_size in (1000, 2000):
            client.log_metric(
                run.info.run_id,
                f"profile/vocabulary_size/{vocabulary_size}/update_ms",
                vocabulary_size / (100.0 if condition.endswith("-FS") else 200.0),
            )
        client.log_metric(run.info.run_id, "profile/points/ok", 2.0)
        artifact_dir = tmp_path / condition
        artifact_dir.mkdir()
        artifact = artifact_dir / "result.json"
        artifact.write_text(
            json.dumps(
                {
                    "schema_name": "ds2-profile",
                    "points": [
                        {
                            "condition_id": condition,
                            "axes": {"vocabulary_size": vocabulary_size},
                            "status": "ok",
                            "metrics": {
                                "update_ms": vocabulary_size
                                / (100.0 if condition.endswith("-FS") else 200.0),
                            },
                        }
                        for vocabulary_size in (1000, 2000)
                    ],
                }
            ),
            encoding="utf-8",
        )
        client.log_artifact(run.info.run_id, str(artifact), "profile")
        client.set_terminated(run.info.run_id, "FINISHED")

    canonical_output = tmp_path / "canonical-results"
    canonical_cache = tmp_path / "canonical-cache"
    write_analysis(
        uri,
        volume=Volume.DS2,
        experiment_ids=["e11"],
        variants=(Variant.IMPLEMENTED,),
        output_dir=canonical_output,
        cache_dir=canonical_cache,
    )
    assert (canonical_output / "ds2_e11_imp.png").exists()
    assert (canonical_output / "ds2_e11_imp_cbow.png").exists()
    scaling = canonical_cache / "render" / "ds2_e11_imp_scaling.csv"
    text = scaling.read_text(encoding="utf-8")
    assert "PF-VSCALE-CBOW-FS" in text
    assert ",2000," in text
    assert (canonical_cache / "analysis_input.json").exists()
    assert (canonical_cache / "prepared" / "e11" / "implemented").exists()
