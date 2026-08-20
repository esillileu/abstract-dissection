from __future__ import annotations

import sys
from pathlib import Path

from exp.framework.execution import RunOptions, RunOrder, RunSelection
from exp.deepscratch.ds1.catalog import ORIGINAL as DS1_ORIGINAL
from exp.deepscratch.ds2.catalog import ORIGINAL as DS2_ORIGINAL
from exp.framework.execution.planning import Planner


def test_default_catalog_sizes_include_ds2_original_word2vec() -> None:
    ds1 = Planner(DS1_ORIGINAL).build(RunSelection(all_experiments=True), RunOptions(device="cpu"))
    ds2 = Planner(DS2_ORIGINAL).build(RunSelection(all_experiments=True), RunOptions(device="cpu"))

    assert len(ds1) == 500
    assert len(ds2) == 162
    assert {plan.experiment_id for plan in ds2} == {
        "e01",
        "e02",
        "e03",
        "e04",
        "e05",
        "e06",
        "e07",
        "e08",
        "e12",
    }


def test_ds2_e05_uses_upstream_better_rnnlm_recipe() -> None:
    spec = DS2_ORIGINAL.load_run_spec(
        Path("exp/deepscratch/ds2/config/original/e05_better_rnnlm.yaml"),
        atomic_run_id="BETTER-RNNLM",
        overrides={},
    ).to_executor_config()

    assert spec["source_experiment"] == "e05"
    assert spec["trial_id"] == "dlfs2.ch06.ptb-better-rnnlm"
    assert spec["training"]["max_epochs"] == 40


def test_atomic_seed_override_and_seed_first_order() -> None:
    plans = Planner(DS1_ORIGINAL).build(
        RunSelection(experiment_ids=("e01",), atomic_run_ids=("OPT-SGD",), seed_values="2,1"),
        RunOptions(device="cpu", overrides={"tracking": {"enabled": False}}, order=RunOrder.SEED_FIRST),
    )

    assert [(plan.atomic_run_id, plan.seed) for plan in plans] == [("OPT-SGD", 2), ("OPT-SGD", 1)]
    spec = DS1_ORIGINAL.load_run_spec(plans[0].path, atomic_run_id=plans[0].atomic_run_id, overrides={"tracking": {"enabled": False}})
    assert spec.to_executor_config()["tracking"]["enabled"] is False


def test_upstream_import_context_restores_modules() -> None:
    from exp.deepscratch.ds1.original.run.common import source_imports

    sentinel = object()
    previous = sys.modules.get("common", sentinel)
    source = Path("exp/deepscratch/ds1/original/source")
    with source_imports(source):
        __import__("common.layers")
        assert "common.layers" in sys.modules
    if previous is sentinel:
        assert "common" not in sys.modules
        assert "common.layers" not in sys.modules
    else:
        assert sys.modules["common"] is previous


def test_ds2_e08_requires_matching_seed_source() -> None:
    spec = DS2_ORIGINAL.load_run_spec(
        Path("exp/deepscratch/ds2/config/original/e08_attention.yaml"),
        atomic_run_id="ATTENTION-ALIGNMENT",
        overrides={},
    ).to_executor_config()

    assert spec["checkpoint"]["source_atomic_run_id"] == "ATTENTION-REVERSE"
    assert spec["checkpoint"]["source_artifact_path"] == "raw/checkpoint.npz"


def test_ds2_e02_expands_original_word2vec_conditions() -> None:
    plans = Planner(DS2_ORIGINAL).build(
        RunSelection(experiment_ids=("e02",)), RunOptions()
    )

    assert len(plans) == 20
    assert {plan.atomic_run_id for plan in plans} == {"PTB-CBOW", "PTB-SKIPGRAM"}


def test_retired_source_tree_result_roots_are_absent() -> None:
    assert not Path("exp/ds1/results/original").exists()
    assert not Path("exp/ds2/results/original").exists()
    assert not Path("exp/deepscratch/ds1/original/legacy_results").exists()
    assert not Path("exp/deepscratch/ds2/original/legacy_results").exists()
