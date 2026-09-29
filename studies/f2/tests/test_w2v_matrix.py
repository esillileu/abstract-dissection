from __future__ import annotations

from pathlib import Path

from f2.definition import DEFINITION
from f2.suites.w2v.readiness import (
    CLASSIFICATIONS,
    DISPOSITIONS,
    RunReportRecord,
    write_readiness_report,
)
from repro_core.execution.definition import RunOptions, RunSelection
from repro_core.execution.planning import Planner


def test_production_w2v_catalogs_cannot_embed_test_scenarios() -> None:
    source_root = Path(__file__).parents[1] / "src/f2/suites"
    assert not [path for path in source_root.rglob("fixtures/*") if path.is_file()]
    for config in source_root.glob("w2v*/config/*.yaml"):
        assert "local-smoke" not in config.read_text()

    tracked = (source_root / "w2v/tracked.py").read_text()
    executor = (source_root / "w2v1/executor.py").read_text()
    for forbidden in (
        "stop_after_epoch",
        "resume_checkpoint",
        "run_local_yaml",
        'status="KILLED"',
    ):
        assert forbidden not in tracked
        assert forbidden not in executor


def test_suite_plans_match_canonical_matrix() -> None:
    canonical_seeds = {1, 7, 19}
    for suite, experiment_id, expected in (
        ("w2v1", "e01", 216),
        ("w2v1", "e02", 18),
        ("w2v2", "e02", 54),
    ):
        plans = Planner(DEFINITION.get_suite(suite)).build(
            RunSelection(experiment_ids=(experiment_id,)), RunOptions()
        )
        assert len(plans) == expected
        assert {plan.experiment_id for plan in plans} == {experiment_id}
        assert {plan.seed for plan in plans} == canonical_seeds


def test_each_available_corpus_uses_only_its_plan_identity() -> None:
    expectations = {
        "w2v1": {
            "wmt--d50-w24m": "w2v1-reconstruction-r2",
            "lm1b--d50-w24m": "w2v1-lm1b-reconstruction-r1",
            "umbc--d50-w24m": "w2v1-umbc-reconstruction-r1",
        },
        "w2v2": {
            "wmt--neg5-no-subsampling": "w2v2-reconstruction-r1",
            "lm1b--neg5-no-subsampling": "w2v2-lm1b-reduced-r1",
            "umbc--neg5-no-subsampling": "w2v2-umbc-reconstruction-r1",
        },
    }
    for suite, variants in expectations.items():
        definition = DEFINITION.get_suite(suite)
        experiment_id = "e01" if suite == "w2v1" else "e02"
        for atomic_run_id, execution_plan_id in variants.items():
            plans = Planner(definition).build(
                RunSelection(
                    experiment_ids=(experiment_id,),
                    atomic_run_ids=(atomic_run_id,),
                    seed_values="1",
                ),
                RunOptions(),
            )
            spec = definition.load_run_spec(
                plans[0].path,
                atomic_run_id=plans[0].atomic_run_id,
                overrides={},
            ).with_seed(1)
            assert spec.identity["execution_plan_id"] == execution_plan_id
            assert "resource_version" not in spec.identity
            assert "corpus_manifest_digest" not in spec.identity


def test_selected_seed_is_part_of_runtime_identity() -> None:
    definition = DEFINITION.get_suite("w2v2")
    spec = definition.load_run_spec(
        definition.config_root / "e02_table3_phrase_skipgram.yaml",
        atomic_run_id="wmt--hs-subsampling",
        overrides={},
    ).with_seed(19)

    assert spec.identity["seed"] == 19
    assert (
        spec.identity["planned_run_slot_id"]
        == "w2v2-reconstruction-r1-hs-subsampling-s19"
    )


def test_target_dispositions_cover_catalog_and_report_is_human_readable(
    tmp_path,
) -> None:
    assert len(DISPOSITIONS) == 13
    assert {item.classification for item in DISPOSITIONS} == {
        "reconstruction",
        "external_baseline",
        "unsupported",
    }
    assert CLASSIFICATIONS == (
        "exact_reproduction",
        "reconstruction",
        "external_baseline",
        "unsupported",
    )
    markdown, csv_path = write_readiness_report(
        tmp_path,
        [
            RunReportRecord(
                2,
                "slot-s1",
                "mlflow-run-1",
                "w2v1-table-2",
                "analogy_accuracy",
                0.2,
                0.1,
                0.3,
                0.8,
                0.232,
                42.0,
                "cpu:model;threads=1",
            )
        ],
    )
    assert "95% CI" in markdown.read_text()
    assert "hardware" in csv_path.read_text().splitlines()[0]
