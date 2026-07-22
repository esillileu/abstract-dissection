from __future__ import annotations

from pathlib import Path
import runpy

import yaml


CONFIG_ROOT = Path("experiments/deepscratch2/config")


def test_deepscratch2_catalog_declares_200_seeded_runs() -> None:
    seeds = yaml.safe_load((CONFIG_ROOT / "seeds.yaml").read_text())["seed_sets"]["research_v1"]["values"]
    configs = [yaml.safe_load(path.read_text()) for path in CONFIG_ROOT.glob("e[0-9][0-9]_*.yaml")]

    assert sum(len(config["variants"]) * config["policy"]["seed_count"] for config in configs) == 200
    assert len(seeds) == 10
    assert all(config["tracking"]["experiment"] == "deepscratch2" for config in configs)


def test_e02_selection_includes_its_shared_and_full_factorial_conditions() -> None:
    run_all = runpy.run_path(Path("experiments/deepscratch2/run_all.py"))
    plans = list(run_all["_plans"](["e02"], [0], "cpu"))

    assert {atomic_run_id for _, atomic_run_id, _, _ in plans} == {
        "W2V-CBOW-FULL",
        "W2V-CBOW-NS",
        "W2V-SG-FULL",
        "W2V-SG-NS",
    }


def test_e01_declares_book_compatible_word2vec_training() -> None:
    config = yaml.safe_load((CONFIG_ROOT / "e01_cbow_skipgram.yaml").read_text())

    assert config["loss"]["reduction"] == "sum"
    assert config["training"]["trainer"] == "book_word2vec"
    assert config["training"]["log_interval"] == 20


def test_e03_better_rnnlm_uses_the_book_learning_rate_decay_divisor() -> None:
    config = yaml.safe_load((CONFIG_ROOT / "e03_rnnlm_comparison.yaml").read_text())

    assert config["scheduler"] == {"name": "validation_decay", "factor": 4.0}


def test_book_baseline_extensions_use_their_declared_toy_settings() -> None:
    word2vec = yaml.safe_load((CONFIG_ROOT / "e06_word2vec_toy_full_softmax.yaml").read_text())
    rnnlm = yaml.safe_load((CONFIG_ROOT / "e07_rnnlm_small_corpus.yaml").read_text())

    assert word2vec["dataset"]["text"] == "You say goodbye and I say hello."
    assert word2vec["training"]["max_epochs"] == 1000
    assert rnnlm["dataset"]["train_limit"] == 1000
    assert rnnlm["loader"] == {"batch_size": 10, "time_size": 5}
    assert rnnlm["optimizer"]["learning_rate"] == 0.1


def test_sequence_books_use_the_original_fixed_split() -> None:
    for name in ("e04_seq2seq_addition.yaml", "e05_attention_seq2seq_date.yaml"):
        dataset = yaml.safe_load((CONFIG_ROOT / name).read_text())["dataset"]
        assert dataset["split_seed"] == 1984
        assert dataset["split_algorithm"] == "legacy_numpy_randomstate"
