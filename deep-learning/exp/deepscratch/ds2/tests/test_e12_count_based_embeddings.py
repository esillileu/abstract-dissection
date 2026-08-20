from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import yaml

from exp.deepscratch.ds2.implemented.spec import parse_run_spec
from exp.deepscratch.ds2.original.run import e12 as original_e12
from exp.deepscratch.ds2.statistical import (
    create_cooccurrence_matrix,
    factorize_ppmi,
    positive_pmi,
)


CONFIG = Path("exp/deepscratch/ds2/config/implemented/e12_count_based_embeddings.yaml")


def test_e12_declares_ppmi_and_both_svd_methods() -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert set(raw["variants"]) == {
        "COUNT-PTB-PPMI",
        "COUNT-PTB-SVD",
        "COUNT-PTB-RANDOMIZED-SVD",
    }
    for atomic_run_id in raw["variants"]:
        spec = parse_run_spec(CONFIG, atomic_run_id=atomic_run_id)
        assert spec.kind == "count_based_embedding"
        assert spec.identity.experiment_id == "e12"
        assert spec.dataset["window_size"] == 2
        assert spec.checkpoint["save_final"] is True
        assert spec.seed_policy["seed_count"] == 10


def test_count_based_pipeline_is_numerically_reusable_and_seeded() -> None:
    corpus = np.array([0, 1, 0, 2, 1, 2], dtype=np.int64)
    cooccurrence = create_cooccurrence_matrix(corpus, 3, 1)
    ppmi = positive_pmi(cooccurrence)
    assert np.array_equal(cooccurrence, cooccurrence.T)
    assert ppmi.dtype == np.float32
    assert np.all(ppmi >= 0)

    full, singular_values, _ = factorize_ppmi(
        ppmi, method="svd", components=2, seed=1
    )
    random_a, random_values_a, right_a = factorize_ppmi(
        ppmi, method="randomized_svd", components=2, seed=1
    )
    random_b, random_values_b, right_b = factorize_ppmi(
        ppmi, method="randomized_svd", components=2, seed=1
    )
    assert full.shape == (3, 2)
    assert singular_values.shape == (3,)
    np.testing.assert_array_equal(random_a, random_b)
    np.testing.assert_array_equal(random_values_a, random_values_b)
    np.testing.assert_array_equal(right_a, right_b)


def test_original_e12_records_reusable_matrices_and_phase_times(
    tmp_path: Path, monkeypatch
) -> None:
    class Util:
        @staticmethod
        def create_co_matrix(corpus, vocab_size, window_size):
            return create_cooccurrence_matrix(corpus, vocab_size, window_size)

        @staticmethod
        def ppmi(matrix, verbose=False):
            del verbose
            return positive_pmi(matrix)

    class Ptb:
        @staticmethod
        def load_data(split):
            assert split == "train"
            return (
                np.array([0, 1, 0, 2, 1, 2], dtype=np.int64),
                {"you": 0, "year": 1, "car": 2},
                {0: "you", 1: "year", 2: "car"},
            )

    util_module = ModuleType("common.util")
    util_module.create_co_matrix = Util.create_co_matrix
    util_module.ppmi = Util.ppmi
    ptb_module = ModuleType("dataset.ptb")
    ptb_module.load_data = Ptb.load_data
    monkeypatch.setitem(sys.modules, "common.util", util_module)
    monkeypatch.setitem(sys.modules, "dataset.ptb", ptb_module)
    monkeypatch.setattr(original_e12, "source_imports", lambda _path: _null_context())
    monkeypatch.setattr(original_e12, "master_seed", lambda: 1)
    monkeypatch.setattr(
        original_e12,
        "randomized_svd",
        lambda matrix, n_components, n_iter, random_state: (
            np.eye(len(matrix), n_components, dtype=np.float32),
            np.ones(n_components, dtype=np.float32),
            np.zeros((n_components, len(matrix)), dtype=np.float32),
        ),
    )

    output = tmp_path / "result"
    original_e12._run("randomized_svd", tmp_path, output, tmp_path)
    with np.load(output / "statistical_matrices.npz") as archive:
        assert {"cooccurrence", "ppmi", "word_vectors", "singular_values", "right_factors"} <= set(archive.files)
    timing = json.loads((output / "timing.json").read_text(encoding="utf-8"))
    assert timing["method"] == "randomized_svd"
    assert timing["total_s"] == (
        timing["cooccurrence_s"]
        + timing["ppmi_s"]
        + timing["decomposition_s"]
    )


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return None
