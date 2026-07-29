from __future__ import annotations

from pathlib import Path

import yaml

from exp.ds2.executor import (
    AttentionAlignmentObservationExecutor,
    LanguageModelExecutor,
    Seq2SeqExecutor,
    Word2VecExecutor,
    _language_model,
    _optimizer,
    _seq_model,
    get_observation_executor,
)
from exp.ds2.spec import parse_run_spec
from mlprosection.core.backend import BackendConfig, make_backend
from mlprosection.nn.model.architecture import (
    CBOW,
    CBOWBatchAdapter,
    SkipGram,
    SkipGramBatchAdapter,
)
from mlprosection.nn.objective import (
    NegativeSampling,
    SoftmaxWithLoss,
    TemporalSoftmaxCrossEntropy,
)

CONFIG_ROOT = Path("exp/ds2/config")


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
                model_type, _adapter = {
                    "CBOW": (CBOW, CBOWBatchAdapter()),
                    "SkipGram": (SkipGram, SkipGramBatchAdapter()),
                }[str(model_values["name"])]
                model = model_type(11, 3, backend=backend)
                objective_type = {
                    "SoftmaxWithLoss": SoftmaxWithLoss,
                    "NegativeSampling": NegativeSampling,
                }[str(config["objective"]["name"])]
                objective = (
                    objective_type(11, backend=backend)
                    if objective_type is NegativeSampling
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
            else:
                executor = get_observation_executor(config)
                assert isinstance(executor, AttentionAlignmentObservationExecutor)
            assert callable(executor.run)
            count += 1
    assert count == 19
