import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

import experiments.cli as cli
from experiments.cli import build_analysis_plans, build_plans, main, parse_overrides, parse_seed_indexes

pytestmark = pytest.mark.legacy


def test_seed_index_ranges_are_deduplicated() -> None:
    assert parse_seed_indexes("0-2,1,4", count=5) == [0, 1, 2, 4]


def test_deepscratch1_plan_selects_one_experiment_and_seed_range() -> None:
    plans = build_plans(
        domain="deepscratch1",
        experiment_ids=["01"],
        all_experiments=False,
        seed_set="research_v1",
        seed_indexes="0",
        device=None,
        overrides={},
    )

    assert [plan.atomic_run_id for plan in plans] == ["TOY-SGD", "TOY-MOM", "TOY-ADAGRAD", "TOY-ADAM"]
    assert {plan.device for plan in plans} == {"cpu"}


def test_deepscratch1_cnn_plans_default_to_gpu() -> None:
    plans = build_plans(
        domain="deepscratch1",
        experiment_ids=["08", "09"],
        all_experiments=False,
        seed_set="research_v1",
        seed_indexes="0",
        device=None,
        overrides={},
    )

    assert {plan.device for plan in plans} == {"cuda:0"}


def test_ds1_plan_selects_gt01_atomic_trials_and_seed_indexes() -> None:
    plans = build_plans(
        domain="ds1",
        experiment_ids=["01"],
        all_experiments=False,
        seed_set="research_v1",
        seed_indexes="1,2,3,4",
        device=None,
        overrides={},
    )

    assert len(plans) == 16
    assert {plan.atomic_run_id for plan in plans} == {
        "MLP-OPT-SGD", "MLP-OPT-MOMENTUM", "MLP-OPT-ADAGRAD", "MLP-OPT-ADAM",
    }
    assert {plan.seed for plan in plans} == {2, 3, 4, 5}
    assert {plan.device for plan in plans} == {"cpu"}


def test_cli_overrides_change_seed_count_and_are_applied_last() -> None:
    overrides = parse_overrides(["policy.seed_count=2", "training.max_epochs=1"])
    plans = build_plans(
        domain="deepscratch2",
        experiment_ids=["01"],
        all_experiments=False,
        seed_set="research_v1",
        seed_indexes=None,
        device="cpu",
        overrides=overrides,
    )

    assert len(plans) == 4
    assert {plan.device for plan in plans} == {"cpu"}


def test_single_execution_mode_does_not_create_seed_repetitions(tmp_path, monkeypatch) -> None:
    config_root = tmp_path / "single" / "config"
    config_root.mkdir(parents=True)
    (config_root / "seeds.yaml").write_text("seed_sets: {research_v1: {values: [10, 20]}}", encoding="utf-8")
    (config_root / "e14_single.yaml").write_text(
        "kind: optimizer_toy\nexecution: {mode: single}\nvariants: {ONLY: {atomic_run_id: ONLY}}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "EXPERIMENTS_ROOT", tmp_path)

    plans = build_plans(
        domain="single",
        experiment_ids=["14"],
        all_experiments=False,
        seed_set="research_v1",
        seed_indexes=None,
        device="cpu",
        overrides={},
    )

    assert len(plans) == 1
    assert plans[0].seed is None


def test_plan_defaults_to_all_experiments(capsys) -> None:
    main(["deepscratch1", "plan"])

    assert "deepscratch1: 465 planned runs" in capsys.readouterr().out


def test_analysis_discovery_uses_each_module_experiment_id() -> None:
    plans = build_analysis_plans(domain="deepscratch2", experiment_ids=["05"])

    assert [(plan.experiment_id, plan.module_name) for plan in plans] == [
        ("e05", "experiments.deepscratch2.analysis.e05_attention_seq2seq_date"),
    ]


def test_analysis_dry_run_lists_selected_script(capsys) -> None:
    main(["deepscratch1", "analyze", "-e", "01", "--dry-run"])

    assert "e01 experiments.deepscratch1.analysis.e01_optimizer_toy" in capsys.readouterr().out


def test_analysis_arguments_are_forwarded(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command, check: seen.append((command, check)))

    cli.run_analysis_plans(
        build_analysis_plans(domain="deepscratch1", experiment_ids=["03"]),
        tracking_uri="http://example.test",
        analysis_args=["--layout", "individual"],
    )

    assert seen[0][0][-4:] == ["--tracking-uri", "http://example.test", "--layout", "individual"]
