from __future__ import annotations

from pathlib import Path

import yaml
from deepscratch.core import BackendConfig, make_backend
from deepscratch.nn.model.architecture import (
    CBOW,
    CBOWBatchAdapter,
    DumbCBOW,
    DumbSkipGram,
    FusedNegativeSamplingCBOW,
    FusedNegativeSamplingSkipGram,
    OneHotCBOW,
    OneHotCBOWBatchAdapter,
    OneHotSkipGram,
    OneHotSkipGramBatchAdapter,
    PairExpandedSkipGramBatchAdapter,
    SkipGram,
    SkipGramBatchAdapter,
)
from deepscratch.nn.objective import (
    FusedNegativeSampling,
    NegativeSampling,
    SoftmaxWithLoss,
    TemporalSoftmaxCrossEntropy,
)

from dlfs.ds2.implemented.executor import (
    AttentionAlignmentObservationExecutor,
    CountBasedEmbeddingExecutor,
    LanguageModelExecutor,
    ProfileExecutor,
    Seq2SeqExecutor,
    Word2VecExecutor,
    _language_model,
    _optimizer,
    _seq_model,
    get_observation_executor,
)
from dlfs.ds2.implemented.spec import parse_run_spec

CONFIG_ROOT = Path("studies/dlfs/src/dlfs/ds2/config/implemented")


def test_e02_declares_gpu_dumb_word2vec_conditions() -> None:
    path = CONFIG_ROOT / "e02_ptb_word2vec.yaml"
    source = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert set(source["variants"]) == {
        "W2V-PTB-CBOW-NS",
        "W2V-PTB-SKIPGRAM-NS",
        "W2V-PTB-CBOW-FULL",
        "W2V-PTB-SKIPGRAM-FULL",
    }
    spec = parse_run_spec(path, atomic_run_id="W2V-PTB-SKIPGRAM-FULL")
    assert {variant["model"]["name"] for variant in source["variants"].values()} == {
        "DumbCBOW",
        "DumbSkipGram",
    }
    assert spec.model["name"] == "DumbSkipGram"
    assert source["execution"]["default_device"] == "cuda:0"
    assert spec.numerics["backend"] == "cupy"
    assert spec.numerics["device"] == "cuda:0"


def test_all_ds2_variants_resolve_and_build_the_declared_components() -> None:
    backend = make_backend(BackendConfig(device="cpu", dtype="float32", seed=1))
    count = 0
    for path in sorted(CONFIG_ROOT.glob("e[0-9][0-9]_*.yaml")):
        source = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(source, dict)
        variants = source["variants"]
        assert isinstance(variants, dict)
        for atomic_run_id in variants:
            config = parse_run_spec(
                path,
                atomic_run_id=atomic_run_id,
            ).to_executor_config()
            model_values = {
                **config["model"],
                "wordvec_size": 3,
                "hidden_size": 4,
                "embedding_size": 3,
            }
            kind = config["kind"]
            if kind == "word2vec":
                representation = str(
                    model_values.get("input_representation", "embedding")
                )
                model_type, _adapter = {
                    ("CBOW", "embedding"): (CBOW, CBOWBatchAdapter()),
                    ("DumbCBOW", "embedding"): (
                        DumbCBOW,
                        CBOWBatchAdapter(),
                    ),
                    ("DumbSkipGram", "embedding"): (
                        DumbSkipGram,
                        PairExpandedSkipGramBatchAdapter(),
                    ),
                    ("FusedNegativeSamplingCBOW", "embedding"): (
                        FusedNegativeSamplingCBOW,
                        CBOWBatchAdapter(),
                    ),
                    ("FusedNegativeSamplingSkipGram", "embedding"): (
                        FusedNegativeSamplingSkipGram,
                        SkipGramBatchAdapter(),
                    ),
                    ("SkipGram", "embedding"): (
                        SkipGram,
                        SkipGramBatchAdapter(),
                    ),
                    ("CBOW", "one_hot"): (
                        OneHotCBOW,
                        OneHotCBOWBatchAdapter(11),
                    ),
                    ("SkipGram", "one_hot"): (
                        OneHotSkipGram,
                        OneHotSkipGramBatchAdapter(11),
                    ),
                }[(str(model_values["name"]), representation)]
                model = model_type(11, 3, backend=backend)
                objective_type = {
                    "SoftmaxWithLoss": SoftmaxWithLoss,
                    "FusedNegativeSampling": FusedNegativeSampling,
                    "NegativeSampling": NegativeSampling,
                }[str(config["objective"]["name"])]
                objective = (
                    objective_type(11, backend=backend)
                    if objective_type
                    in {
                        FusedNegativeSampling,
                        NegativeSampling,
                    }
                    else objective_type(backend=backend)
                )
                _optimizer(config, model, objective)
                executor = Word2VecExecutor()
            elif kind == "language_modeling":
                model = _language_model(
                    str(model_values["name"]),
                    11,
                    model_values,
                    backend,
                )
                objective = TemporalSoftmaxCrossEntropy(backend=backend)
                _optimizer(config, model, objective)
                executor = LanguageModelExecutor()
            elif kind == "seq2seq":
                model = _seq_model(
                    str(model_values["name"]),
                    11,
                    model_values,
                    backend,
                )
                objective = TemporalSoftmaxCrossEntropy(backend=backend)
                _optimizer(config, model, objective)
                executor = Seq2SeqExecutor()
            elif kind == "observation":
                executor = get_observation_executor(config)
                assert isinstance(executor, AttentionAlignmentObservationExecutor)
            elif kind == "count_based_embedding":
                executor = CountBasedEmbeddingExecutor()
            else:
                assert kind == "performance_profile"
                executor = ProfileExecutor()
            assert callable(executor.run)
            count += 1
    assert count == 55
