from __future__ import annotations

from f2.definition import DEFINITION
from f2.suites.w2v.matrix import (
    CANONICAL_SEEDS,
    canonical_slots,
    planned_slot_id,
    w2v1_conditions,
    w2v2_conditions,
)
from f2.suites.w2v.readiness import (
    CLASSIFICATIONS,
    DISPOSITIONS,
    RunReportRecord,
    write_readiness_report,
)
from repro_core.execution.definition import RunOptions, RunSelection
from repro_core.execution.planning import Planner


def test_canonical_matrix_has_explicit_cost_and_approval_policy() -> None:
    w2v1 = canonical_slots("w2v1-reconstruction-r2", w2v1_conditions())
    w2v2 = canonical_slots("w2v2-reconstruction-r1", w2v2_conditions())

    assert len(w2v1) == 24 * len(CANONICAL_SEEDS)
    assert len(w2v2) == 6 * len(CANONICAL_SEEDS)
    assert len({slot["planned_run_slot_id"] for slot in (*w2v1, *w2v2)}) == 90
    assert all(slot["requires_approval"] for slot in (*w2v1, *w2v2))
    assert all(slot["estimated_token_updates"] > 0 for slot in (*w2v1, *w2v2))


def test_suite_plans_match_canonical_matrix() -> None:
    for suite, expected in (("w2v1", 72), ("w2v2", 18)):
        plans = Planner(DEFINITION.get_suite(suite)).build(
            RunSelection(all_experiments=True), RunOptions()
        )
        canonical = [plan for plan in plans if plan.atomic_run_id != "local-smoke"]
        assert len(canonical) == expected
        assert {plan.seed for plan in canonical} == set(CANONICAL_SEEDS)


def test_selected_seed_is_part_of_runtime_identity() -> None:
    definition = DEFINITION.get_suite("w2v2")
    spec = definition.load_run_spec(
        definition.config_root / "e01_phrase_skipgram.yaml",
        atomic_run_id="hs-subsampling",
        overrides={},
    ).with_seed(19)

    assert spec.identity["seed"] == 19
    assert spec.identity["planned_run_slot_id"] == planned_slot_id(
        "w2v2-reconstruction-r1", "hs-subsampling", 19
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
