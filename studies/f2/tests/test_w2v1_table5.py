from __future__ import annotations

from pathlib import Path

from f2.catalog.manifest import read_manifest
from f2.definition import DEFINITION
from repro_core.execution.definition import RunOptions, RunSelection
from repro_core.execution.planning import Planner


def test_table5_plan_and_catalog_slots() -> None:
    definition = DEFINITION.get_suite("w2v1")
    plans = Planner(definition).build(
        RunSelection(experiment_ids=("04",)), RunOptions()
    )
    assert len(plans) == 18
    catalog = read_manifest(Path("studies/f2/catalog/w2v.json"))
    slots = {slot["planned_run_slot_id"]: slot for slot in catalog["planned_run_slots"]}
    for plan in plans:
        spec = definition.load_run_spec(
            plan.path, atomic_run_id=plan.atomic_run_id, overrides={}
        ).with_seed(plan.seed)
        corpus, condition = plan.atomic_run_id.split("--")
        architecture = condition.split("-", 1)[0]
        config = spec.to_executor_config()
        identity = config["identity"]
        assert identity["planned_run_slot_id"] in slots
        slot = slots[identity["planned_run_slot_id"]]
        assert slot["seed"] == plan.seed
        assert slot["atomic_run_id"] == condition
        assert slot["plan_experiment_id"].endswith(identity["experiment_spec_id"])
        assert identity["study"] == "table5"
        assert identity["execution_plan_id"] == f"w2v1-table5-{corpus}-r1"
        assert config["vocabulary"]["max_lexical_words"] == 1_000_000
        training = config["training"]
        assert training["model_kind"] == (
            "cbow" if architecture == "cbow" else "skip_gram"
        )
        assert training["window_radius"] == (4 if architecture == "cbow" else 10)
        assert training["context_policy"] == (
            "fixed" if architecture == "cbow" else "dynamic"
        )
        assert training["epochs"] == 1
        assert training["objective_kind"] == "hierarchical_softmax"
        assert training["initial_learning_rate"] == 0.025
        assert training["subsampling_threshold"] == 0.0
