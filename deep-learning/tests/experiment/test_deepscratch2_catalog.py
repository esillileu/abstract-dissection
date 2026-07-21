from __future__ import annotations

from pathlib import Path

import yaml


CONFIG_ROOT = Path("experiments/deepscratch2/config")


def test_deepscratch2_catalog_declares_140_seeded_runs() -> None:
    seeds = yaml.safe_load((CONFIG_ROOT / "seeds.yaml").read_text())["seed_sets"]["research_v1"]["values"]
    configs = [yaml.safe_load(path.read_text()) for path in CONFIG_ROOT.glob("e[0-9][0-9]_*.yaml")]

    assert sum(len(config["variants"]) * config["policy"]["seed_count"] for config in configs) == 140
    assert len(seeds) == 10
    assert all(config["tracking"]["experiment"] == "deepscratch2" for config in configs)
