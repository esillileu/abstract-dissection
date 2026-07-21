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


def test_e01_declares_book_compatible_word2vec_training() -> None:
    config = yaml.safe_load((CONFIG_ROOT / "e01_cbow_skipgram.yaml").read_text())

    assert config["loss"]["reduction"] == "sum"
    assert config["training"]["trainer"] == "book_word2vec"
    assert config["training"]["log_interval"] == 20


def test_sequence_books_use_the_original_fixed_split() -> None:
    for name in ("e04_seq2seq_addition.yaml", "e05_attention_seq2seq_date.yaml"):
        dataset = yaml.safe_load((CONFIG_ROOT / name).read_text())["dataset"]
        assert dataset["split_seed"] == 1984
        assert dataset["split_algorithm"] == "legacy_numpy_randomstate"
