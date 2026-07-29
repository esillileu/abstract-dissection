from __future__ import annotations

import pytest

from exp.ds2.executor import _apply_validation_decay, _optimizer
from exp.ds2.spec import parse_run_spec
from mlprosection import Tensor
from mlprosection.nn.types import Parameter
from mlprosection.optim.transform import ClipGradNorm


class _SingleParameterModel:
    def __init__(self) -> None:
        self.parameter = Parameter(Tensor([0.0], backend="cpu"))

    def named_parameters(self):
        return (("weight", self.parameter),)


def test_word2vec_runspec_connects_source_curve_contract() -> None:
    spec = parse_run_spec("exp/ds2/config/e02_ptb_word2vec.yaml", atomic_run_id="W2V-PTB-CBOW-NS")

    assert spec.identity.group_id == "GT02"
    assert spec.source_curve is not None
    assert spec.source_curve.kind == "interval_mean_loss"
    assert spec.source_curve.every_updates == 20
    assert spec.model["name"] == "CBOW"
    assert spec.objective["name"] == "NegativeSampling"
    assert spec.objective["book_reduction"] == "sum_terms_mean_examples"


def test_toy_skipgram_matches_cbow_conditions_except_architecture() -> None:
    cbow = parse_run_spec(
        "exp/ds2/config/e01_toy_word2vec.yaml",
        atomic_run_id="W2V-TOY-CBOW-FULL",
    )
    skipgram = parse_run_spec(
        "exp/ds2/config/e01_toy_word2vec.yaml",
        atomic_run_id="W2V-TOY-SKIPGRAM-FULL",
    )

    assert skipgram.model["name"] == "SkipGram"
    assert skipgram.objective["name"] == "SoftmaxWithLoss"
    assert skipgram.identity.structure_signature == "toy-skipgram-full-softmax-w1-e5"
    assert skipgram.dataset == cbow.dataset
    assert skipgram.loader == cbow.loader
    assert skipgram.optimizer == cbow.optimizer
    assert skipgram.budget == cbow.budget
    assert skipgram.model["embedding_size"] == cbow.model["embedding_size"]


def test_lm_recipe_runspec_declares_valid_selected_checkpoint_inputs() -> None:
    spec = parse_run_spec("exp/ds2/config/e05_ptb_lm_recipes.yaml", atomic_run_id="LM-BETTER-RECIPE")

    assert spec.identity.group_id == "GT05"
    assert spec.checkpoint["save_best"] is True
    assert spec.evaluations[0].axis == "epoch"
    assert spec.evaluations[0].sources == ("valid",)
    assert spec.evaluations[1].axis == "terminal"
    assert spec.evaluations[1].sources == ("test",)


@pytest.mark.parametrize(
    "atomic_run_id",
    ["LM-RNN-RECIPE", "LM-LSTM-RECIPE", "LM-BETTER-RECIPE"],
)
def test_lm_recipe_validation_decay_applies_to_every_model(
    atomic_run_id: str,
) -> None:
    spec = parse_run_spec(
        "exp/ds2/config/e05_ptb_lm_recipes.yaml",
        atomic_run_id=atomic_run_id,
    )
    optimizer = _optimizer(
        spec.config, _SingleParameterModel(), _SingleParameterModel()
    )

    _apply_validation_decay(spec.config, optimizer)

    assert optimizer.lr == 5.0


def test_seq2seq_runspec_declares_predictions_and_attention_observation_inputs() -> None:
    spec = parse_run_spec("exp/ds2/config/e07_date_seq2seq.yaml", atomic_run_id="SEQD-ATTN-REV")

    assert spec.identity.group_id == "GT07"
    assert spec.recording["predictions"] == {"split": "test", "count": 10, "decode": "greedy"}
    assert spec.recording["attention"] == {"selection_seed": 1984, "count": 5}
    assert spec.model["name"] == "AttentionSeq2seq"


def test_ds2_runspec_allows_device_timing_profiling_policy() -> None:
    spec = parse_run_spec(
        "exp/ds2/config/e01_toy_word2vec.yaml",
        atomic_run_id="W2V-TOY-CBOW-FULL",
        overrides={"profiling": {"device_timing": True}},
    )

    assert spec.profiling["device_timing"] is True
    assert spec.config["profiling"] == {"enabled": False, "device_timing": True}


def test_gt03_catalog_excludes_the_custom_loop_variant() -> None:
    with pytest.raises(ValueError, match="unknown atomic_run_id"):
        parse_run_spec(
            "exp/ds2/config/e03_small_rnnlm.yaml",
            atomic_run_id="LM-SMALL-RNN-CUSTOM",
        )


def test_ds2_runspec_still_rejects_legacy_evaluation_key() -> None:
    with pytest.raises(ValueError, match="old catalog keys"):
        parse_run_spec(
            "exp/ds2/config/e01_toy_word2vec.yaml",
            atomic_run_id="W2V-TOY-CBOW-FULL",
            overrides={"evaluation": {"test_at_end": True}},
        )


@pytest.mark.parametrize(
    ("config_path", "atomic_run_id", "expected_max_norm"),
    [
        ("exp/ds2/config/e01_toy_word2vec.yaml", "W2V-TOY-CBOW-FULL", None),
        ("exp/ds2/config/e03_small_rnnlm.yaml", "LM-SMALL-RNN", None),
        ("exp/ds2/config/e06_addition_seq2seq.yaml", "SEQA-VAN-FWD", 5.0),
    ],
)
def test_ds2_optimizer_owns_group_gradient_clipping(
    config_path: str,
    atomic_run_id: str,
    expected_max_norm: float | None,
) -> None:
    spec = parse_run_spec(config_path, atomic_run_id=atomic_run_id)

    optimizer = _optimizer(
        spec.config, _SingleParameterModel(), _SingleParameterModel()
    )
    clipping = [
        hook for hook in optimizer.pre_step_hooks if isinstance(hook, ClipGradNorm)
    ]

    if expected_max_norm is None:
        assert clipping == []
    else:
        assert len(clipping) == 1
        assert clipping[0].max_norm == expected_max_norm
