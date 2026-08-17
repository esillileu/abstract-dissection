from __future__ import annotations

from pathlib import Path

from mlflow.tracking import MlflowClient

from exp.deepscratch.ds2.catalog import IMPLEMENTED
from exp.deepscratch.ds2.implemented.spec import parse_run_spec
from exp.deepscratch.ds2.profile.e11.render import render
from exp.framework.execution import RunOptions, RunSelection
from exp.framework.execution.planning import Planner


CONFIG = Path(
    "exp/deepscratch/ds2/config/implemented/"
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
    assert (
        config["tracking"]["tags"]["profile.study_kind"]
        == "axis_scaling"
    )


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
                "result.durable_complete": "true",
                "result.schema.name": "ds2-profile",
                "protocol.version": "ds2-e11-vocabulary-size-scaling-v1",
            },
        )
        for vocabulary_size in (1000, 2000):
            client.log_metric(
                run.info.run_id,
                f"profile/vocabulary_size/{vocabulary_size}/update_ms",
                vocabulary_size / (100.0 if condition.endswith("-FS") else 200.0),
            )
        client.set_terminated(run.info.run_id, "FINISHED")

    png, csv, markdown = render(uri, tmp_path / "results")
    assert png.exists()
    assert csv.exists()
    assert markdown.exists()
    text = csv.read_text(encoding="utf-8")
    assert "implemented-cbow-fs" in text
    assert ",2000," in text
    assert "vocabulary-size scaling" in markdown.read_text(encoding="utf-8")
