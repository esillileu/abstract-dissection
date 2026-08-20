from __future__ import annotations

from pathlib import Path

import yaml

from exp.deepscratch.ds1.implemented.executor import (
    SupervisedClassificationExecutor,
    _model,
    _objective,
    _optimizer,
    _training_parameters,
    get_observation_executor,
)
from exp.deepscratch.ds1.implemented.spec import parse_run_spec


CONFIG_ROOT = Path("exp/deepscratch/ds1/config/implemented")


def test_all_ds1_variants_resolve_and_build_the_declared_components() -> None:
    count = 0
    for path in sorted(CONFIG_ROOT.glob("e[0-9][0-9]_*.yaml")):
        source = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(source, dict)
        variants = source["variants"]
        assert isinstance(variants, dict)
        for atomic_run_id in variants:
            config = parse_run_spec(path, atomic_run_id=atomic_run_id).to_executor_config()
            if config["kind"] == "supervised_classification":
                model = _model(config["model"])
                objective = _objective(config["objective"], model.backend)
                _optimizer(
                    config["optimizer"],
                    _training_parameters(model, objective),
                )
                executor = SupervisedClassificationExecutor()
            else:
                executor = get_observation_executor(config)
            assert callable(executor.run)
            count += 1
    assert count == 68
