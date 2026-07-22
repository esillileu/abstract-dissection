from __future__ import annotations

import pytest

from exp.ds2.spec import parse_run_spec


def test_word2vec_runspec_connects_source_curve_contract() -> None:
    spec = parse_run_spec("exp/ds2/config/e02_ptb_word2vec.yaml", atomic_run_id="W2V-PTB-CBOW-NS")

    assert spec.identity.group_id == "GT02"
    assert spec.source_curve is not None
    assert spec.source_curve.kind == "interval_mean_loss"
    assert spec.source_curve.every_updates == 20
    assert spec.model["architecture"] == "cbow"
    assert spec.model["objective"] == "negative_sampling"


def test_lm_recipe_runspec_declares_valid_selected_checkpoint_inputs() -> None:
    spec = parse_run_spec("exp/ds2/config/e05_ptb_lm_recipes.yaml", atomic_run_id="LM-BETTER-RECIPE")

    assert spec.identity.group_id == "GT05"
    assert spec.checkpoint["save_best"] is True
    assert spec.evaluations[0].axis == "epoch"
    assert spec.evaluations[0].sources == ("valid",)
    assert spec.evaluations[1].axis == "terminal"
    assert spec.evaluations[1].sources == ("test",)


def test_seq2seq_runspec_declares_predictions_and_attention_observation_inputs() -> None:
    spec = parse_run_spec("exp/ds2/config/e07_date_seq2seq.yaml", atomic_run_id="SEQD-ATTN-REV")

    assert spec.identity.group_id == "GT07"
    assert spec.recording["predictions"] == {"split": "test", "count": 10, "decode": "greedy"}
    assert spec.recording["attention"] == {"selection_seed": 1984, "count": 5}
    assert spec.model["alias"] == "AttentionSeq2seq"


def test_ds2_runspec_allows_device_timing_profiling_policy() -> None:
    spec = parse_run_spec(
        "exp/ds2/config/e01_toy_word2vec.yaml",
        atomic_run_id="W2V-TOY-CBOW-FULL",
        overrides={"profiling": {"device_timing": True}},
    )

    assert spec.profiling["device_timing"] is True
    assert spec.config["profiling"] == {"enabled": False, "device_timing": True}


def test_ds2_runspec_still_rejects_legacy_evaluation_key() -> None:
    with pytest.raises(ValueError, match="old catalog keys"):
        parse_run_spec(
            "exp/ds2/config/e01_toy_word2vec.yaml",
            atomic_run_id="W2V-TOY-CBOW-FULL",
            overrides={"evaluation": {"test_at_end": True}},
        )
