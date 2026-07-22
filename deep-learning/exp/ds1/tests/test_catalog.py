from __future__ import annotations

from pathlib import Path

import yaml

from exp.ds1.executor import _model
from exp.ds1.spec import parse_run_spec


CONFIG_ROOT = Path("exp/ds1/config")


def test_all_ds1_variants_resolve_and_build_the_declared_model() -> None:
    count = 0
    for path in sorted(CONFIG_ROOT.glob("e[0-9][0-9]_*.yaml")):
        source = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(source, dict)
        variants = source["variants"]
        assert isinstance(variants, dict)
        for atomic_run_id in variants:
            config = parse_run_spec(path, atomic_run_id=atomic_run_id).to_executor_config()
            if config["kind"] == "supervised_classification":
                _model(config["model"])
            count += 1
    assert count == 65
